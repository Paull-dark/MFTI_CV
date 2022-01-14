import pickle

import pandas as pd
import numpy as np
from typing import Optional
import pickle
from glob import glob
from pathlib import Path
from tqdm import tqdm
from detectron2.structures import BoxMode

import cv2

from config import DATA_ROOT, DATA_ROOT_RESIZED

def build_COVID19_data_dicts(
        imdir: Path,
        train_df: pd.DataFrame,
        use_cache: bool = True,
        target_indices: Optional[np.ndarray] = None,
        debug: bool = False,
        data_type: str='train',
        cache_mode: str='train'
):
    cache_path = Path('.') / f"dataset_dicts_cache_{cache_mode}.pkl"
    if not use_cache or not cache_path.exists():
        print('Creating the data ==> ...')
        df_meta = pd.read_csv(f'{DATA_ROOT_RESIZED}/meta_1.csv')
        sub_meta = df_meta[df_meta.split == data_type]
        if debug:
            sub_meta = sub_meta.iloc[:100] # for debuging

        # load one image to get image size
        image_id = sub_meta.iloc[0,0]
        image_path = f'{DATA_ROOT_RESIZED}/train/{image_id}.jpg'
        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, sub_meta_row in tqdm(sub_meta.iterrows(), total=len(sub_meta)):
            record = {}
            image_id, height, width,s = sub_meta_row.values
            filename = f'{DATA_ROOT_RESIZED}/train/{image_id}.jpg'
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            objs = []
            if data_type != 'train':
                dataset_dicts.append(record)
            else:
                for index2, row in train_df.query('id==@image_id').iterrows():
                    #print(row)
                    #print(row["class_name"])
                    #class_name = row["class_name"] #class
                    class_id = row["integer_label"]
                    if class_id == 2: #"no class or none"
                        # Используем класс none  c bbox покрывающей всю площадь изобр.
                        bbox_resized = [0, 0, resized_width, resized_height]
                        #bbox_resized = [50,50,200,200] !!!!!!!!!!!!!!!!!!!!!!!!!1
                        obj = {
                            "bbox": bbox_resized,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": class_id,
                        }
                    else:
                        # bbox_original = [int(row["x_min"]), int(row["y_min"]), int(row["x_max"]), int(row["y_max"])]
                        h_ratio = resized_height / height
                        w_ratio = resized_width / width
                        bbox_resized = [
                            float(row["x_min"]) * w_ratio,
                            float(row["y_min"]) * h_ratio,
                            float(row["x_max"]) * w_ratio,
                            float(row["y_max"]) * h_ratio,
                        ]
                        obj = {
                            "bbox": bbox_resized,
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": class_id,
                        }
                        objs.append(obj)
                record['annotations'] = objs
                dataset_dicts.append(record)
        with open(cache_path, mode='wb') as f:
            pickle.dump(dataset_dicts,f)

    print(f"Load from cache {cache_path}")
    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    if target_indices is not None:
        dataset_dicts = [dataset_dicts[i] for i in target_indices]
    return dataset_dicts



def build_COVID19_data_dicts_test(
    imdir: Path,
    #test_meta: pd.DataFrame,
    use_cache: bool = True,
    debug: bool = False,
    cache_mode: str = 'test'
):

    debug_str = f"_debug{int(debug)}"
    cache_path = Path('.') / f"dataset_dicts_cache_{cache_mode}.pkl"
    if not use_cache or not cache_path.exists():
        print('Creating the data ==> ...')
        df_meta = pd.read_csv(f'{DATA_ROOT_RESIZED}/meta_1.csv')
        test_meta=df_meta[df_meta.split=="test"]
        if debug:
            test_meta = test_meta.iloc[:10]  # For debug....
        # Load 1 image to get image size.
        image_id = test_meta.iloc[0,0]
        #image_path = str(imgdir / "test" / f"{image_id}.jpg")
        image_path = f'{DATA_ROOT_RESIZED}/test/{image_id}.jpg'

        image = cv2.imread(image_path)
        resized_height, resized_width, ch = image.shape
        print(f"image shape: {image.shape}")

        dataset_dicts = []
        for index, test_meta_row in tqdm(test_meta.iterrows(), total=len(test_meta)):
            record = {}

            image_id, height, width,s = test_meta_row.values
            filename = f'{DATA_ROOT_RESIZED}/test/{image_id}.jpg'
            record["file_name"] = filename
            record["image_id"] = image_id
            record["height"] = resized_height
            record["width"] = resized_width
            dataset_dicts.append(record)
        with open(cache_path, mode="wb") as f:
            pickle.dump(dataset_dicts, f)

    with open(cache_path, mode="rb") as f:
        dataset_dicts = pickle.load(f)
    return dataset_dicts