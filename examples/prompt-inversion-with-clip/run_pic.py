import os
import dataclasses
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from rlprompt.trainers import TrainerConfig, make_trainer_for_clip
from rlprompt.modules import QLModuleConfig, make_ql_module
from rlprompt.models import (LMModelConfig, SinglePromptModelConfig,
                             make_lm_model, make_lm_adaptor_model, 
                             make_single_prompt_model)
from rlprompt.utils.utils import (colorful_print, compose_hydra_config_store,
                                  get_hydra_output_dir)
from pic_helpers import (ClipRewardConfig, make_prompted_clip_reward)

# Compose default config
config_list = [ClipRewardConfig, LMModelConfig, 
               SinglePromptModelConfig, QLModuleConfig, 
               TrainerConfig]
cs = compose_hydra_config_store('base_pic', config_list)


@hydra.main(version_base=None, config_path="./", config_name="pic_config")
def main(config: "DictConfig"):
    colorful_print(OmegaConf.to_yaml(config), fg='red')
    output_dir = get_hydra_output_dir()
    if config.adaptor_model:
        policy_model = make_lm_adaptor_model(config)
    else:
        policy_model = make_lm_model(config)
    prompt_model = make_single_prompt_model(policy_model, config)
    # config.style_classifier = get_style_classifier('train', config)
    reward = make_prompted_clip_reward(config)
    algo_module = make_ql_module(prompt_model, reward, config)

    config.save_dir = os.path.join(output_dir, config.save_dir)
    trainer = make_trainer_for_clip(algo_module, config)
    trainer.train(config=config)

if __name__ == "__main__":
    main()
