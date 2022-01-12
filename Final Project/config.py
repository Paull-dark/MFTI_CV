from dataclasses import dataclass, field
from typing import Dict

TRAIN_STUDY_CSV = './dataset/siim-covid19-detection/train_study_level.csv'
TRAIN_IMAGE_CSV = './dataset/siim-covid19-detection/train_image_level.csv'
DATA_ROOT = './dataset/siim-covid19-detection'
DATA_ROOT_RESIZED = './dataset/SIIM-COVID19-Resized'

@dataclass
class Flags:
    # General
    debug: bool = True
    outdir: str = './output/'

    #data_config
    root = './dataset/siim-covid19-detection'
    imgdir_name: str = DATA_ROOT_RESIZED

    split_mode: str = "all_train"  # all_train or valid20

    # Training config
    iter: int = 10000
    warm_up_iters: int = 1000
    ims_per_batch: int = 2  # images per batch, this corresponds to "total batch size"
    num_workers: int = 4
    lr_scheduler_name: str = "WarmupMultiStepLR"  # WarmupMultiStepLR (default) or WarmupCosineLR
    base_lr: float = 0.00025
    roi_batch_size_per_image: int = 512
    eval_period: int = 10000
    aug_kwargs: Dict = field(default_factory=lambda: {})

    def update(self, param_dict: Dict) -> "Flags":
        # Overwrite by `param_dict`
        for key, value in param_dict.items():
            if not hasattr(self, key):
                raise ValueError(f"[ERROR] Unexpected key for flag = {key}")
            setattr(self, key, value)
        return self
