import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Dict, Any, Union, List
import os
import wandb
import json
import click

from rlprompt.modules import BaseModule
from rlprompt.utils import utils
from .trainer_utils import get_default_train_op, set_random_seed


class Trainer_for_clip:
    """Trainer that runs for a specified number of epochs. 

    Each epoch can run for a specified number of batches.
    Evaluation is done at the end of each epoch """

    def __init__(
        self,
        module: BaseModule,
        # Train params
        num_train_epochs: int,
        max_train_steps: int,
        # Eval params
        do_eval: bool,
        eval_batch_size: int,
        eval_steps: int,
        # Save params
        do_save: bool,
        save_dir: str,
        save_steps: int,
        # Optimizer params
        learning_rate: float,
        gradient_clip: bool,
        gradient_clip_norm: float,
        # Checkpoint params
        checkpoint_path: Optional[str],
        # Random seed
        random_seed: Optional[int],
        # Wandb reporting
        report_to_wandb: bool,
        project_name: Optional[str],
        num_exp: Optional[int]
        # run_name: Optional[str]
    ):

        self.module = module
        self.num_train_epochs = num_train_epochs
        self.max_train_steps = max_train_steps

        self.do_eval = do_eval
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps

        self.do_save = do_save
        self.save_dir = save_dir
        self.save_steps = save_steps

        self.train_op = get_default_train_op(self.module._model,
                                             learning_rate,
                                             gradient_clip,
                                             gradient_clip_norm)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

        if random_seed is not None:
            set_random_seed(random_seed)

        self.report_to_wandb = report_to_wandb
        self.project_name = project_name
        # self.run_name = run_name
        self.num_exp = num_exp

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.module.load_state_dict(checkpoint["model_state_dict"])
        print(click.style(f"Loaded module from {checkpoint_path}", fg="green"))


    def _train_step(
        self,
        step: int,
        batch: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        model = self.module.train()
        # sync_target_model
        model._pre_steps(step)

        loss, batch_log = model(batch, step=step)
        if loss is None: return batch_log
        loss.backward()
        self.train_op()

        return batch_log

    def train(self,
              report_to_wandb: Optional[bool] = None,
              project_name: Optional[str] = None,
            #   run_name: Optional[str] = None,
              config: Optional["DictConfig"] = None) -> None:
        # Configure Wandb reporting
        if report_to_wandb is None:
            report_to_wandb = self.report_to_wandb
        if project_name is None:
            project_name = self.project_name
        # if run_name is None: 
        #     run_name = self.run_name
        if config is not None: 
            config = eval(str(config))
        if report_to_wandb:
            run = wandb.init(project=project_name, config=config)
            text_table = wandb.Table(columns=["epoch", "score", "prompt"])
            # wandb.watch(self.module, log=None)

        # Create saving path
        eval_save_dir = os.path.join(self.save_dir, "eval")
        ckpt_save_dir = os.path.join(self.save_dir, "ckpt")
        if not os.path.exists(eval_save_dir):
            os.makedirs(eval_save_dir)
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        # Determine whether to evaluate by epoch or steps
        eval_by_steps = self.eval_steps > 0
        # Determine whether to save by epoch or steps
        save_by_steps = self.save_steps > 0
        
        total_steps = 0
        # make dummy batch
        # batch['source_texts'] is required in "single_prompt_model.generate" 
        batch = {'source_texts': ['<pad>'] * 32}
        for epoch in range(self.max_train_steps):
            batch_log = self._train_step(total_steps, batch)
            if report_to_wandb:
                wandb.log(batch_log)
            total_steps += 1
            if self.do_eval and eval_by_steps \
                and total_steps % self.eval_steps == 0:  
                    text_table.add_data(epoch, round(self.module._reward.max_reward, 3),
                                        self.module._reward.max_prompt)
            # if self.do_eval and eval_by_steps \
            #         and total_steps % self.eval_steps == 0:
            #     output_save_path = \
            #         os.path.join(eval_save_dir,
            #                         f'outputs.step.{total_steps}.json')
            #     eval_prompt, eval_log = self.evaluate(output_save_path=output_save_path)
            #     text_table.add_data(epoch, round(eval_log['eval/score'], 2), eval_prompt)
            #     if report_to_wandb:
            #         wandb.log(eval_log)

            # if self.do_save and (self.save_steps > 9000) and total_steps % self.save_steps == 0:
            #     torch.save({"steps": total_steps,
            #                 "model_state_dict": self.module.state_dict()},
            #                 os.path.join(ckpt_save_dir,
            #                             f"{self.num_exp}_ckpt.step.{total_steps}.pth")) 


            if self.do_eval and not eval_by_steps:
                # output_save_path = os.path.join(eval_save_dir,
                #                                 f'outputs.epoch.{epoch+1}.json')
                text_table.add_data(epoch, round(self.model._reward.max_reward, 3),
                    self.model._reward.max_prompt)

            # if self.do_save and not save_by_steps:
            #     torch.save({"steps": total_steps,
            #                 "model_state_dict": self.module.state_dict()},
            #                os.path.join(ckpt_save_dir,
            #                             f"ckpt.epoch.{epoch+1}.pth"))

        run.log({"trainig_samples": text_table})
        df = text_table.get_dataframe()
        csv_dir = os.path.join(self.save_dir, f'{self.num_exp}_prompt.csv')
        df.to_csv(csv_dir, encoding='utf-8')    
        wandb.finish() 
    
    @torch.no_grad()
    # check the input of reward
    def evaluate(
        self,
        output_save_path: Optional[str] = None,
        compute_scores: bool = True
    ) -> Dict[str, np.number]:
        # make empty batch
        batch = {'source_texts': ['<pad>'] * 32}
        model = self.module.eval()
        hypos = []
        infer_outputs = model.infer(batch)
        hypos += infer_outputs['sample_tokens']
        score, score_log = model.compute_rewards(
            batch=batch,
            output_tokens=infer_outputs['sample_tokens'])
        
        # if output_save_path is not None:
        #     json.dump({'output_tokens': hypos,
        #                'scores': scores},
        #               open(output_save_path, 'w'))
        
        score = score.mean().item()
        utils.add_prefix_to_dict_keys_inplace(
            score_log,
            prefix=f"eval/rewards/")
        print('Finish Eval')
        
        prompt = self.module._model._model.tokenizer.convert_tokens_to_string(hypos[-1])
        return prompt, \
               utils.unionize_dicts([
                score_log,
                {f"eval/score": score }
            ])
