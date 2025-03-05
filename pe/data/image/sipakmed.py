from pathlib import Path
import random
import cv2
import pandas as pd
import os
from tqdm import tqdm, trange
import numpy as np
import zipfile
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import Counter, defaultdict

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.util import download
from datasets import load_dataset

FRAC_ATLAS_LABEL_NAMES = [
    0,
    1,
    2,
    3,
    4,
]


class Sipakmed(Data):
    """The Cat dataset."""

    def __init__(self, root_dir="data", res=512, limit=-1):
        """Constructor.

        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 512
        :type res: int, optional
        """
        cache_path = Path(root_dir)/f'.cache_{limit}'
        print(cache_path)
        if not os.path.exists(cache_path):
        
        # if limit > 0:
        #     dataset['train'] = dataset['train'].select(random.sample(list(range(len(dataset['train']))), k=limit))
            images = []
            labels = []
            shapes = defaultdict(int)
            image_size = 256
            transformations = []
            transformations.extend(
                [transforms.Resize(image_size), transforms.CenterCrop(image_size)]
            )

            transformations.extend([transforms.ToTensor()])

            dataset = ImageFolder(
                root=root_dir, transform=transforms.Compose(transformations)
            )
            for tensor, label in tqdm(dataset, desc="Converting to numpy"):
                images.append(tensor.numpy())
                labels.append(label)
            # assert False
            
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
