import torch
import copy
from typing import Optional, List, Dict, Any, Union, Tuple

from rlprompt.models import BaseModel
from rlprompt.modules import BaseModule
from rlprompt.rewards import BaseReward
from rlprompt.modules.module_utils import ForwardMode, get_reward_shaping_func
from rlprompt.modules.replay_buffer import SimpleReplayTokenSeqPool
from rlprompt.losses import sql_loss_with_sparse_rewards
from rlprompt.utils import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SQLModule(BaseModule):
    def __init__(
        self,
        model: BaseModel,
        reward: Optional[BaseReward],
        sql_loss_impl: str,
        training_mode: str,
        mix_strategy: Optional[str],
        off_train_batch_size: Optional[int],
        target_update_method: str,
        target_update_steps: Optional[int],
        target_learning_rate: float,
        reward_shaping: bool,
        reward_shaping_old_min: float,
        reward_shaping_old_max: float,
        reward_shaping_new_min: float,
        reward_shaping_new_max: float,
        gamma: float,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
    ):
        super().__init__()
        # Initialize self._model and self._reward
        assert target_update_method in ["copy", "polyak"]
        assert not (top_k is not None and top_p < 1.0), \
               "Only one of top_k or top_p should be selected"

        self._model = model
        # if target_model is None:
        #     self._target_model = copy.deepcopy(self._model)
        # else:
        #     self._target_model = target_model
        # for p1, p2 in zip(self._model.parameters(), self._target_model.parameters()):
        #     if p1.data.ne(p2.data).sum() > 0:
        #         print(False)
        #     print(True) 
        self._reward = reward
        
        ###################################
        if training_mode == 'sql-onpolicy':
            self._replay_buffer = None
        else:
            self._replay_buffer = SimpleReplayTokenSeqPool(max_size=64000, 
                                                       max_seq_len=self._model.prompt_length, 
                                                       state_dim=self._model.model_dim)
        self.off_train_batch_size = off_train_batch_size
        self.num_trajs = 0
        #################################
    
        self._sql_loss_impl = sql_loss_impl
        self._training_mode = training_mode
        self._mix_strategy = mix_strategy
        self._forward_modes = _get_forward_modes(training_mode, mix_strategy)
        self._target_update_method = target_update_method
        self._target_update_steps = target_update_steps
        self._target_learning_rate = target_learning_rate
        self._top_k = top_k
        self._top_p = top_p
        self._num_beams = num_beams
        self._gamma = gamma

        if reward_shaping is True:
            self._reward_shaping_func = get_reward_shaping_func(
                old_min=reward_shaping_old_min,
                old_max=reward_shaping_old_max,
                new_min=reward_shaping_new_min,
                new_max=reward_shaping_new_max)
        else:
            self._reward_shaping_func = lambda _r: _r

    def _sync_target_model(self) -> None:
        if self._target_update_method == "copy":
            self._model._model.sync_target_model(target_learning_rate=1.)
        else:
            self._model._model.sync_target_model(self._target_learning_rate)

    def _pre_steps(self, step: int) -> None:
        if self._target_update_method == "polyak":
            self._sync_target_model()
        elif self._target_update_method == "copy" \
                and step % self._target_update_steps == 0:
            self._sync_target_model()

    def forward(self, batch: Dict[str, Any], step=None) -> Tuple[Union[torch.Tensor, Dict],
                                                      Dict[str, Any]]:
        loss_list = []
        loss_log_list = []
        for mode in self._forward_modes:
            _loss, _loss_log = self._forward(mode=mode, batch=batch)
            loss_list.append(_loss)
            loss_log_list.append(_loss_log)

        # Only exploration 
        if None in loss_list:
            return None, utils.unionize_dicts(loss_log_list)
        # https://discuss.pytorch.org/t/get-the-mean-from-a-list-of-tensors/31989/2
        loss = torch.mean(torch.stack(loss_list))
        loss_log = utils.unionize_dicts(loss_log_list)

        return loss, loss_log

    def _forward(
        self,
        mode: ForwardMode,
        batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict]:
        # if mode != ForwardMode.SQL_ON and mode != ForwardMode.INFER:
        #     # TODO: Enable training modes other than on-policy
        #     raise NotImplementedError('Only on-policy sampling and greedy '
        #                               'inference is supported now')

        if mode == ForwardMode.SQL_ON:
            (states, logits, output_tokens, output_ids, sequence_lengths) = self._decode_sampling(batch=batch)
            self.num_trajs += states.shape[0]
            raw_rewards, rewards_log = \
            self.compute_rewards(batch=batch, 
                                  output_tokens=output_tokens,
                                  mode="train")
            shaped_rewards = self._reward_shaping_func(raw_rewards)
            logits_ = self._model._model.get_values(states=states, target=True)
            
        elif mode == ForwardMode.SQL_OFF:
            # exploration here
            (states, logits, output_tokens, output_ids, sequence_lengths) = self._decode_sampling(batch=batch)
                
            raw_rewards, rewards_log = \
            self.compute_rewards(batch=batch, 
                                  output_tokens=output_tokens,
                                  mode="train")
            shaped_rewards = self._reward_shaping_func(raw_rewards)
            
            self._replay_buffer.add_samples({
                'input_ids': output_ids.cpu().numpy(), # [batch_size, seq_len]
                'states': states.cpu().numpy(),        # [batch_size, seq_len, dim]
                'reward': shaped_rewards.cpu().numpy(),# [batch_size]
                'sequence_lengths': sequence_lengths.cpu().numpy()  # [batch_size] 
            })
            self.num_trajs += states.shape[0]
            if self._replay_buffer.size < self.off_train_batch_size:
                return None, {f"{mode.value}/num_traj": int(self.num_trajs) }
                
            # sample from replay buffer
            _b = self._replay_buffer.random_batch(batch_size=self.off_train_batch_size)            
            states = torch.tensor(_b["states"]).float().to(device)
            output_ids = torch.tensor(_b["input_ids"]).long().to(device)
            shaped_rewards = torch.tensor(_b['reward']).float().to(device)
            sequence_lengths = torch.tensor(_b["sequence_lengths"]).long().to(device)
            logits = self._model._model.get_values(states=states, target=False)
            logits_ = self._model._model.get_values(states=states, target=True)
        
        # loss 
        sql_loss, sql_loss_log = sql_loss_with_sparse_rewards(
            implementation=self._sql_loss_impl,
            logits=logits,
            logits_=logits_,
            actions=output_ids,
            sampled_actions=None,
            rewards=shaped_rewards,
            sequence_length=sequence_lengths)

        utils.add_prefix_to_dict_keys_inplace(
            rewards_log, prefix=f"{mode.value}/rewards/")
        utils.add_prefix_to_dict_keys_inplace(
            sql_loss_log, prefix=f"{mode.value}/")
        sql_loss_log = utils.unionize_dicts([
            rewards_log,
            sql_loss_log,
            {
                f"{mode.value}/rewards_raw": raw_rewards.mean().cpu().numpy(),
                f"{mode.value}/rewards_shaped": shaped_rewards.mean().cpu().numpy(),
                f"{mode.value}/num_traj": int(self.num_trajs)
            },
        ])

        return sql_loss, sql_loss_log

    def compute_rewards(
        self,
        batch: Dict[str, Any],
        output_tokens: List[List[str]],
        to_tensor: bool = True,
        mode: str = "infer"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        
        if self._reward.task_lm == 'clip':
            prompt_strs = [self._model._model.tokenizer.convert_tokens_to_string(s)
                           for s in output_tokens] 
            rewards_tensor, rewards_log = self._reward(
                prompt_strs=prompt_strs,
                to_tensor=to_tensor,
                mode=mode)
        else:
            rewards_tensor, rewards_log = \
                self._reward(**batch, output_tokens=output_tokens, 
                             to_tensor=to_tensor, mode=mode)
        
        rewards_tensor = rewards_tensor.to(device)            
        return rewards_tensor, rewards_log

    def infer(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, Union[torch.Tensor, torch.LongTensor, List[List[str]]]]:
        return self._model.generate(**batch,
                                    do_sample=False,
                                    top_k=self._top_k,
                                    top_p=self._top_p,
                                    num_beams=self._num_beams,
                                    infer=True)

    def _decode_sampling(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]],
               torch.LongTensor, torch.LongTensor]:
        outputs = self._model.generate(**batch,
                                       do_sample=True,
                                       top_k=self._top_k,
                                       top_p=self._top_p,
                                       num_beams=self._num_beams)

        batch_ = {k: v for k, v in batch.items()}
        # batch_.update(outputs)

        # outputs_ = self._model.teacher_forcing(**batch_)
        # with torch.no_grad():
        #     outputs_ = self._model._model.get_values(states=outputs["states"],
        #                                             target=True)

        return (outputs['states'],
                outputs['sample_logits'].contiguous(),
                outputs['sample_tokens'],
                outputs['sample_ids'].contiguous(),
                outputs['sample_lengths'].contiguous())


def _get_forward_modes(
    training_mode: str,
    mix_strategy: Optional[str]
) -> List[ForwardMode]:
    if training_mode == "sql-mixed":
        candidate_modes = [
            ForwardMode.SQL_OFF,
            ForwardMode.SQL_ON]

        if mix_strategy == "alternate":
            modes = [candidate_modes[step % len(candidate_modes)]]
        elif mix_strategy == "mix":
            modes = candidate_modes

    else:
        training_mode_map = {"sql-onpolicy": ForwardMode.SQL_ON,
                             "sql-offpolicy": ForwardMode.SQL_OFF}

        modes = [training_mode_map[training_mode]]

    return modes
