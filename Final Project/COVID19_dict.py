import pandas as pd
import numpy as np
from typing import Optional
from glob import glob
from pathlib import Path

import cv2

from config import DATA_ROOT, DATA_ROOT_RESIZED

def build_COVID19_data_dicts(
        imdir: Path,
        train_df: pd.DataFrame,
        use_cache: bool = True,
        target_indeces: Optional[np.ndarray] = None,
        debug: bool = False,
        data_type: str='train'
):
    cache_path = Path('.') / f"dataset_dicts_cache_{data_type}.pkl"
    if not use_cache or not cache_path.exists():
        print('Creating the data ==> ...')
        df_meta = pd.read_csv(f'{DATA_ROOT_RESIZED}/img_sz_640/meta_sz_640.csv')
        train_meta = df_meta[df_meta.split == 'train']
        if debug:
            train_meta = train_meta.iloc[:100] # for debuging

        # load one image to get image size
        image_id = train_meta.iloc[0,0]
        image_path = f'{DATA_ROOT_RESIZED}/img_sz_640/train/{image_id}.jpg'
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")




df = build_COVID19_data_dicts()
print(len(df))