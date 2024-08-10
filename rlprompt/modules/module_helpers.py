from dataclasses import dataclass
from typing import Optional

from rlprompt.modules import SQLModule, SPARSE_QL_Module
from rlprompt.models import BaseModel
from rlprompt.rewards import BaseReward

def make_ql_module(model: BaseModel,
                    reward: BaseReward,
                    config: "DictConfig",
                    target_model: Optional[BaseModel] = None) -> SQLModule:
    if config.algo == 'sql':
        return SQLModule(model, reward, 
                        config.loss_impl, config.training_mode, 
                        config.mix_strategy, config.off_train_batch_size, 
                        config.target_update_method, 
                        config.target_update_steps, config.target_learning_rate, 
                        config.reward_shaping, 
                        config.reward_shaping_old_min, 
                        config.reward_shaping_old_max, 
                        config.reward_shaping_new_min, 
                        config.reward_shaping_new_max, 
                        config.gamma,
                        config.top_k, config.top_p, config.num_beams)
    elif config.algo =="sparse-ql":
        return SPARSE_QL_Module(model, reward,
                                config.loss_impl, config.training_mode,
                                config.mix_strategy, config.off_train_batch_size, 
                                config.target_update_method, 
                                config.target_update_steps, config.target_learning_rate, 
                                config.reward_shaping, 
                                config.reward_shaping_old_min, 
                                config.reward_shaping_old_max, 
                                config.reward_shaping_new_min, 
                                config.reward_shaping_new_max,
                                config.gamma, 
                                config.top_k, config.top_p, config.num_beams)  
        

@dataclass
class QLModuleConfig:
    algo: str = "sql" # "sparse-ql"
    loss_impl: str = "v2_v2r_v3_v3r" 
    training_mode: str = "sql-onpolicy"
    off_train_batch_size: int = 64
    mix_strategy: Optional[str] = None
    # Target model setting
    target_update_method: str = "polyak"
    target_update_steps: Optional[int] = None
    target_learning_rate: float = 0.001
    # Reward shaping linearly transforms reward range of [old_min, old_max]
    # to [new_min, new_max]
    reward_shaping: bool = True
    reward_shaping_old_min: float = 0
    reward_shaping_old_max: float = 100
    reward_shaping_new_min: float = -10
    reward_shaping_new_max: float = 10
    gamma: float = 1.
    # Prompt generation setting
    top_k: Optional[int] = None
    top_p: Optional[float] = 1.0
    num_beams: int = 1
