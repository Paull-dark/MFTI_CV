import os
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from med_vis import dicom2array
import config


def resize(array,size,keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size,size), resample)

    else:
        im = im.resize((size,size), resample)

    return im

train = pd.read_csv(config.TRAIN_IMAGE_CSV)


image_id = []
dim0 = []
dim1 = []
splits = []

for split in ['test', 'train']:
    save_dir = f'./dataset/resized_pics/{split}/'

    os.makedirs(save_dir, exist_ok=True)
    for dirname, _, filenames in tqdm(os.walk(f'{config.DATA_ROOT}/{split}')):
        for file in filenames:
            # set keep_ratio=True to have original aspect ratio
            xray = dicom2array(os.path.join(dirname, file))
            im = resize(xray, size=1024)
            im.save(os.path.join(save_dir, file.replace('dcm', 'jpg')))

            image_id.append(file.replace('.dcm', ''))
            dim0.append(xray.shape[0])
            dim1.append(xray.shape[1])
            splits.append(split)

df = pd.DataFrame.from_dict({'image_id': image_id, 'dim0': dim0, 'dim1': dim1, 'split': splits})
df.to_csv('meta_1.csv', index=False)