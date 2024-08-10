from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, List
from clip_reward import ClipReward

# Key: (dataset, dataset_seed, split)
def make_prompted_clip_reward(
        config: "DictConfig") -> ClipReward:
    return ClipReward(config.task_lm, config.clip_pretrain, config.target_image_idx, config.dataset_name,
)


@dataclass
class ClipRewardConfig:
    task_lm: str = "Vit-H-14"
    clip_pretrain: str = "laion2b_s32b_b79k"
    target_image_idx: Optional[int] = -1
    dataset_name: Optional[str]="celeba"
