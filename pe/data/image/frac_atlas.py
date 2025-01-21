from pathlib import Path
import random
import cv2
import pandas as pd
import os
from tqdm import tqdm, trange
import numpy as np
import zipfile
from PIL import Image
import torchvision.transforms as T
from collections import Counter, defaultdict

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.util import download
from datasets import load_dataset

FRAC_ATLAS_LABEL_NAMES = [
    0,
    1,
]


class FracAtlas(Data):
    """The Cat dataset."""

    def __init__(self, root_dir="data", res=512, limit=-1):
        """Constructor.

        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 512
        :type res: int, optional
        """
        cache_path = Path(root_dir)/'.cache'
        if not os.path.exists(cache_path):
            data_files = {"train": os.path.join(root_dir, "**")}
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
            )
            if limit > 0:
                dataset['train'] = dataset['train'].select(random.sample(list(range(len(dataset['train']))), k=limit))
            transform = T.Resize(res)
            images = []
            labels = []
            shapes = defaultdict(int)
            target_height = 624
            target_width = 512
            for i in trange(len(dataset['train']), desc="Processing private images.."):
                    label = dataset['train'][i]['label']
                    image = dataset['train'][i]['image']
                    image = image.convert('RGB')
                    image = transform(image)
                    image = np.array(image)
                    current_height, current_width = image.shape[:2]
                    if current_width > current_height:
                        image = np.rot90(image)
                        current_height, current_width = image.shape[:2]
                    
                    if target_height != current_height or target_width != current_width:
                        # Calculate scaling factor to maintain aspect ratio
                        height_ratio = target_height / current_height
                        width_ratio = target_width / current_width
                        scaling_factor = min(height_ratio, width_ratio)
                        
                        # Resize image while maintaining aspect ratio
                        new_height = int(current_height * scaling_factor)
                        new_width = int(current_width * scaling_factor)
                        resized_image = cv2.resize(image, (new_width, new_height))
                        
                        # Create a black canvas of target size
                        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                        
                        # Calculate padding positions to center the image
                        y_offset = (target_height - new_height) // 2
                        x_offset = (target_width - new_width) // 2
                        
                        # Place the resized image in the center of the canvas
                        padded_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
                        image = padded_image
                    
                    shapes[image.shape] += 1
                    # print(image.shape)
                    images.append(image)
                    labels.append(label)
                    
            print('Shapes', dict(Counter(shapes).most_common()))
            data_frame = pd.DataFrame(
                {
                    IMAGE_DATA_COLUMN_NAME: images,
                    LABEL_ID_COLUMN_NAME: labels,
                }
            )
            print(f"Writing to cache: {cache_path}")
            data_frame.to_pickle(cache_path)
        else:
            print(f"Reading from cache: {cache_path}")
            data_frame = pd.read_pickle(cache_path)
        metadata = {"label_info": [{"name": n} for n in FRAC_ATLAS_LABEL_NAMES]}
        super().__init__(data_frame=data_frame, metadata=metadata)