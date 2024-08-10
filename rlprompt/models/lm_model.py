import torch
from torch import nn
import numpy as np
from typing import Optional, List, Dict, Union

from transformers import pipeline, AutoTokenizer
import copy
from .base_model import BaseModel
from .base_model import SUPPORTED_LMS, LM_HIDDEN_SIZES, BOS_TOKENS, PAD_TOKENS
from .model_utils import _top_k_logits, _top_p_logits

class LM_Model(BaseModel):
    """

    """
    def __init__(
        self,
        # MLP-specific parameters
        policy_lm: str,
        hidden_size: int,
        logit_bias: bool,
        fluent: bool,
        fluent_top_k: Optional[int],
        # Generation parameters
        max_decoding_length: int,
        eos_token_id: Optional[int]
    ):
        super().__init__()

        assert policy_lm in SUPPORTED_LMS  # TODO: Support more LMs
        model = policy_lm
        self.model_name = policy_lm
        self.device = 0  # TODO
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            pad_token=PAD_TOKENS[policy_lm])
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False

        self.logit_bias = logit_bias
        self.fluent = fluent
        self.fluent_top_k = fluent_top_k
        self.max_decoding_length = max_decoding_length
        self.eos_token_id = eos_token_id

        self.model_dim = LM_HIDDEN_SIZES[model]
        self.vocab_size = self.tokenizer.vocab_size
        self.mlp = _build_one_layer_mlp(in_dim=self.model_dim,
                                        out_dim=self.vocab_size,
                                        hidden_size=hidden_size).to(self.device)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(_init_weights)

        self.target_mlp = copy.deepcopy(self.mlp)
        # self.sync_target_model()

    def _mlp_forward(self, state: torch.Tensor, target=False) -> torch.Tensor:
        if target:
            with torch.no_grad():
                logits = self.target_mlp(state)
        else: 
            logits = self.mlp(state)

        if self.fluent:
            lm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k)
            min_values: torch.Tensor = values[:, -1].unsqueeze(-1)
            logits = torch.where(lm_logits < min_values,
                                 torch.full_like(logits, float('-inf')),
                                 logits)
        return logits
    # not use
    def teacher_forcing(
        self,
        source_texts: List[str],
        sample_ids: torch.Tensor,
        states: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        state, past_key_values = self._get_generation_cache(source_texts)
        sample_logits = []
        for i in range(sample_ids.shape[-1]):
            logits = self._mlp_forward(state, target=True)
            logits = logits + self.logit_bias

            actions = sample_ids[:, i]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            sample_logits.append(logits.unsqueeze(dim=1))
            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_logits = torch.cat(sample_logits, dim=1)
        output = dict(sample_logits=sample_logits,
                      sample_ids=sample_ids)
        return output

    def get_values(
        self,
        states: torch.Tensor,
        target: bool,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        states : [batch_size, seq_len, dim]
        """
        assert states.dim() == 3
        logits_ = self._mlp_forward(states, target=target) # [batch_size, seq_len, vocab]
        if self.logit_bias:
            lm_logits = self.generator.model.lm_head(states)
            return logits_ + lm_logits
        return logits_
        
    def sample(
        self,
        source_texts: List[str],
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
        eos_token_id: Optional[int],
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        states, sample_ids, sample_logits = [], [], []
        for i in range(max_new_tokens):
            logits = self._mlp_forward(state)  # [batch_size, vocab_size]
            # logits = logits + self.logit_bias
            if self.logit_bias:
                lm_logits = self.generator.model.lm_head(state)
                sampling_logits = logits + lm_logits
            # print(logits[:, 4:].min().item(), logits.max().item())
            if top_k is not None:
                sampling_logits = _top_k_logits(logits, k=top_k)
            elif top_p is not None:
                sampling_logits = _top_p_logits(logits, p=top_p)
            else:
                sampling_logits = logits

            actions = (torch.distributions.categorical
                       .Categorical(logits=sampling_logits)
                       .sample())  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            
            states.append(state.unsqueeze(dim=1))        # [batch_size, 1, dim]
            sample_ids.append(actions.unsqueeze(dim=1))  # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1))# [batch_size, 1, vocab_size]

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        # [batch_size, prompt_length, dim]
        states = torch.cat(states, dim=1)
        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(states=states,
                      sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def greedy_search(self,
                      source_texts: List[str],
                      max_new_tokens: Optional[int],
                      eos_token_id: Optional[int],
                      **kwargs):
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._mlp_forward(state)
            logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())

            actions = logits.argmax(dim=-1)  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def _get_generation_cache(self,
                              source_texts: List[str],
                              past_key_values=None):
        token_encoding = (self.generator
                          .tokenizer(source_texts,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt')
                          .to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        if 'opt' in self.model_name:
            outputs = self.generator.model.model(input_ids,
                                                past_key_values=past_key_values,
                                                use_cache=True)
            
        else:
            outputs = self.generator.model.transformer(input_ids,
                                                       past_key_values=past_key_values,
                                                       use_cache=True)
        last_token_hidden_state = \
            outputs.last_hidden_state[np.arange(input_ids.shape[0]),
                                      (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        return last_token_hidden_state, past_key_values

    def generate(
        self,
        source_texts: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        assert num_beams == 1, "Beam search not supported yet"
        if max_new_tokens is None:
            max_new_tokens = self.max_decoding_length
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        is_greedy_gen_mode = (do_sample == False) and (num_beams == 1)
        is_sample_gen_mode = (do_sample == True) and (num_beams == 1)
        assert is_greedy_gen_mode or is_sample_gen_mode

        if is_greedy_gen_mode:
            return self.greedy_search(source_texts=source_texts,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id)
        elif is_sample_gen_mode:
            return self.sample(source_texts=source_texts,
                               top_k=top_k,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=eos_token_id)

    def sync_target_model(self, target_learning_rate=1.) -> None:
        # "copy" target_learning_rate =1.         
        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        # if target_update_method == "copy":
            # self.target_mlp.load_state_dict(self.mlp.state_dict())

        # Target network update
        # if target_update_method == "polyak":
        for param_, param in zip(self.target_mlp.parameters(),
                                    self.mlp.parameters()):
            param_.data.copy_((1 - target_learning_rate) * param_ + target_learning_rate * param)

##########################
# For SPARSE-QL

MIN_LOGIT_VALUE = -50000.
def sparse_max_normalized(
    logits: torch.Tensor,
    with_filter: bool = True,
    top_k: int = 10000
) -> torch.Tensor:
    """
    Arguments:
        logits: (bs, vocab_size), (bs, seq_len, vocab_size)
    
    Returns:
        spmax: (bs, seq_len)
    """
    # assert not (logits < MIN_LOGIT_VALUE).any()
    if logits.dim() == 2:
        logits = logits[:, None, :]
    vocab_size = logits.shape[-1]
    device = logits.device
    values, indices = torch.sort(logits)
    cumsum_values = values.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
    mask = torch.arange(1, vocab_size+1).flip(dims=[-1])[None, None, ...].to(device) * values + 1 > cumsum_values
    # if with_filter:
        # assert ((top_k + 1) > mask.sum(-1)).all()
    supp_z = mask * values
    tau_z = (supp_z.sum(-1) - 1.) / mask.sum(-1)
    spmax_z = supp_z ** 2 - tau_z.unsqueeze(-1) ** 2
    spmax_z = (spmax_z * mask).sum(-1) + 1
    # return tau_z, spmax_z / 2, mask.sum(-1).float()
    return tau_z

class LM_SPARSE_Model(BaseModel):
    """
    """
    def __init__(
        self,
        # MLP-specific parameters
        policy_lm: str,
        hidden_size: int,
        logit_bias: bool,
        fluent: bool,
        fluent_top_k: Optional[int],
        # Generation parameters
        max_decoding_length: int,
        eos_token_id: Optional[int]
    ):
        super().__init__()

        assert policy_lm in SUPPORTED_LMS  # TODO: Support more LMs
        model = policy_lm
        self.device = 0  # TODO
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            pad_token='<|endoftext|>')
        self.generator = pipeline("text-generation",
                                  tokenizer=self.tokenizer,
                                  model=model,
                                  device=self.device)
        for param in self.generator.model.parameters():
            param.requires_grad = False

        self.logit_bias = logit_bias
        self.fluent = fluent
        self.fluent_top_k = fluent_top_k
        self.max_decoding_length = max_decoding_length
        self.eos_token_id = eos_token_id

        self.model_dim = LM_HIDDEN_SIZES[model]
        self.vocab_size = self.tokenizer.vocab_size
        self.mlp = _build_one_layer_mlp(in_dim=self.model_dim,
                                        out_dim=self.vocab_size,
                                        hidden_size=hidden_size).to(self.device)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.0001)
                m.bias.data.fill_(-0.0001)
        self.mlp.apply(_init_weights)
        self.target_mlp = copy.deepcopy(self.mlp)
        # self.sync_target_model()

    def _mlp_forward(self, state: torch.Tensor, target=False) -> torch.Tensor:
        if target:
            with torch.no_grad():
                logits = self.target_mlp(state)
        else: 
            logits = self.mlp(state)
        
        if self.fluent :
            lm_logits = self.generator.model.lm_head(state)
            values, _ = torch.topk(lm_logits, k=self.fluent_top_k) # [bs, seq_len, top_k], or [bs, top_k]
            min_values = values[..., -1].unsqueeze(-1) # [bs, seq_len, 1] or [bs, 1]
            assert (MIN_LOGIT_VALUE < min_values).all()
            try:
                logits = torch.where(lm_logits < min_values,
                                        torch.full_like(logits, MIN_LOGIT_VALUE),
                                        logits)
            except:
                import pdb; pdb.set_trace()
            return logits, ~(lm_logits < min_values)

        return logits, torch.ones_like(logits)

    def _get_normalized_prob(self, logits, mask):
        # norm, spmax, sp_num_v = sparse_max_normalized(logits, with_filter=self.fluent, top_k=self.fluent_top_k)
        norm = sparse_max_normalized(logits, with_filter=self.fluent, top_k=self.fluent_top_k)
        probs = torch.clamp(logits - norm, min=0.)
        if (probs - (probs * mask)).any():
            import pdb; pdb.set_trace()
        # return probs, spmax, sp_num_v
        return probs
    
        
    def teacher_forcing(
        self,
        source_texts: List[str],
        sample_ids: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        state, past_key_values = self._get_generation_cache(source_texts)

        sample_logits, sample_masks = [], []
        for i in range(sample_ids.shape[-1]):
            logits, mask = self._mlp_forward(state)
            # logits = logits + self.logit_bias

            actions = sample_ids[:, i]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            sample_logits.append(logits.unsqueeze(dim=1))
            sample_masks.append(mask)
            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_logits = torch.cat(sample_logits, dim=1)
        output = dict(sample_logits=sample_logits,
                      sample_masks=sample_masks,
                      sample_ids=sample_ids)
        return output

    def get_values(
        self,
        states: torch.Tensor,
        target: bool,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        states : [batch_size, seq_len, dim]
        """
        assert states.dim() == 3
        logits, masks = self._mlp_forward(states, target=target) # [batch_size, seq_len, vocab]
        # return dict(sample_logits=logits, )
        return logits, masks
        
    def sample(
        self,
        source_texts: List[str],
        top_k: Optional[int],
        top_p: float,
        max_new_tokens: Optional[int],
        eos_token_id: Optional[int],
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        states, sample_ids, sample_logits =[], [], [] 
        sample_masks = []
        # logit_spmaxs, num_vs = [], [] 
        for i in range(max_new_tokens):
            logits, mask = self._mlp_forward(state)  # [batch_size, vocab_size]
            # logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())            
            # logits to probs
            # probs, spmax, num_v = self._get_normalized_prob(logits, mask)
            probs = self._get_normalized_prob(logits, mask)

            # if top_k is not None:
                # sampling_logits = _top_k_logits(logits, k=top_k)
            # elif top_p is not None:
                # sampling_logits = _top_p_logits(logits, p=top_p)
            # else:
                # sampling_logits = probs

            actions = (torch.distributions.categorical
                       .Categorical(probs=probs)
                       .sample())  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            states.append(state.unsqueeze(dim=1))        # [batch_size, 1, dim]
            sample_ids.append(actions.unsqueeze(dim=1))  # [batch_size, 1]
            sample_logits.append(logits.unsqueeze(dim=1))# [batch_size, 1, vocab_size]
            sample_masks.append(mask.unsqueeze(dim=1))
            # logit_spmaxs.append(spmax.unsqueeze(dim=1))
            # num_vs.append(num_v.unsqueeze(dim=1))
            
            # [batch_size, 1, vocab_size]

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        # [batch_size, prompt_length, dim]
        states = torch.cat(states, dim=1)
        # [batch_size, prompt_length]
        sample_ids = torch.cat(sample_ids, dim=1)
        # [batch_size, prompt_length, vocab_size]
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_masks = torch.cat(sample_masks, dim=1)
        # logit_spmaxs = torch.cat(logit_spmaxs, dim=1)
        # num_vs = torch.cat(num_vs, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(states=states,
                      sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_masks=sample_masks,
                      sample_ids=sample_ids,
                    #   logit_spmaxs=logit_spmaxs.squeeze(-1),
                    #   num_vs=num_vs.squeeze(-1),
                      sample_lengths=sample_lengths)
        return output

    def greedy_search(self,
                      source_texts: List[str],
                      max_new_tokens: Optional[int],
                      eos_token_id: Optional[int],
                      **kwargs):
        if eos_token_id is not None:
            raise NotImplementedError(
                "Only support fixed length prompt for now")

        state, past_key_values = self._get_generation_cache(source_texts)
        sample_tokens = [[] for _ in source_texts]
        sample_ids, sample_logits = [], []
        for i in range(max_new_tokens):
            logits = self._mlp_forward(state)[0]
            # logits = logits + self.logit_bias
            # print(logits[:, 4:].min().item(), logits.max().item())

            actions = logits.argmax(dim=-1)  # [batch_size]
            tokens = [self.generator.tokenizer.convert_ids_to_tokens([a])[0]
                      for a in actions.tolist()]
            token_strs = [self.generator.tokenizer.convert_tokens_to_string([t])
                          for t in tokens]

            for s, t in zip(sample_tokens, tokens): 
                s.append(t)
            sample_ids.append(actions.unsqueeze(dim=1))
            sample_logits.append(logits.unsqueeze(dim=1))

            state, past_key_values = self._get_generation_cache(token_strs,
                                                                past_key_values)

        sample_ids = torch.cat(sample_ids, dim=1)
        sample_logits = torch.cat(sample_logits, dim=1)
        sample_lengths = (torch.tensor([max_new_tokens
                                        for _ in range(sample_ids.shape[0])])
                          .to(self.device))

        output = dict(sample_tokens=sample_tokens,
                      sample_logits=sample_logits,
                      sample_ids=sample_ids,
                      sample_lengths=sample_lengths)
        return output

    def _get_generation_cache(self,
                              source_texts: List[str],
                              past_key_values=None):
        token_encoding = (self.generator
                          .tokenizer(source_texts,
                                     padding=True,
                                     truncation=True,
                                     return_tensors='pt')
                          .to(self.device))
        input_ids = token_encoding['input_ids']
        input_lengths = token_encoding['attention_mask'].sum(dim=1)
        outputs = self.generator.model.transformer(input_ids,
                                                   past_key_values=past_key_values,
                                                   use_cache=True)
        last_token_hidden_state = \
            outputs.last_hidden_state[np.arange(input_ids.shape[0]),
                                      (input_lengths - 1)]
        past_key_values = outputs.past_key_values
        return last_token_hidden_state, past_key_values

    def generate(
        self,
        source_texts: List[str],
        do_sample: bool,
        top_k: Optional[int],
        top_p: float,
        num_beams: int,
        max_new_tokens: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        assert num_beams == 1, "Beam search not supported yet"
        if max_new_tokens is None:
            max_new_tokens = self.max_decoding_length
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        is_greedy_gen_mode = (do_sample == False) and (num_beams == 1)
        is_sample_gen_mode = (do_sample == True) and (num_beams == 1)
        assert is_greedy_gen_mode or is_sample_gen_mode

        if is_greedy_gen_mode:
            return self.greedy_search(source_texts=source_texts,
                                      max_new_tokens=max_new_tokens,
                                      eos_token_id=eos_token_id)
        elif is_sample_gen_mode:
            return self.sample(source_texts=source_texts,
                               top_k=top_k,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               eos_token_id=eos_token_id)

    def sync_target_model(self, target_learning_rate=1.) -> None:
        # "copy" target_learning_rate =1.         
        # https://github.com/transedward/pytorch-dqn/blob/master/dqn_learn.py#L221
        # if target_update_method == "copy":
            # self.target_mlp.load_state_dict(self.mlp.state_dict())

        # Target network update
        # if target_update_method == "polyak":
        for param_, param in zip(self.target_mlp.parameters(),
                                    self.mlp.parameters()):
            param_.data.copy_((1 - target_learning_rate) * param_ + target_learning_rate * param)

def _build_one_layer_mlp(in_dim, out_dim, hidden_size):
    W1 = nn.Linear(in_dim, hidden_size)
    A1 = nn.ReLU()
    W2 = nn.Linear(hidden_size, out_dim)
    return nn.Sequential(W1, A1, W2)
