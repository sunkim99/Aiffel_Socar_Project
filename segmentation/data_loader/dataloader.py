import os, glob, cv2, random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A

from tqdm.notebook import tqdm
from PIL import Image
from albumentations.pytorch import transforms
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


class Custom_dataset(torch.utils.data.Dataset):
    """
    Description
     : A class to make a customized dataset.

    Parameters
     : data_dir : a directory that includes image folder and label folder.
     : transform : Albumentations compose.

    """
    def __init__(self, data_dir: str, transform: A.Compose = None):

        self.transform = transform

        # example ) './dataset/dent/train/'
        self.data_dir = data_dir
        self.image_dir = os.path.join(self.data_dir, 'images')
        self.label_dir = os.path.join(self.data_dir, 'masks')
        
        lst_image = os.listdir(self.image_dir)
        lst_label = os.listdir(self.label_dir)
        
        # self.image_dir / self.label_dir 내 모든 파일경로 리스트로 저장
        self.lst_label = lst_label
        self.lst_image = lst_image
        
    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index: int):
        
        # example ) 20190421_7010_22161407_34d91d2eb5440e51ff9ffcae578be67e.jpg -> 20190421_7010_22161407_34d91d2eb5440e51ff9ffcae578be67e
        name = self.lst_image[index].split('.')[0]

        # Image dtype : uint8 / label dtype : uint8
        image = np.array(Image.open(glob.glob(self.image_dir + f'/{name}*')[0])).astype('uint8')
        label = Image.open(glob.glob(self.label_dir + f'/{name}*')[0])

        # Let label images have only 2 values.
        label = (np.array(label.convert('L')) != 0).astype('uint8')

        # Sometimes, image shape is not aligned with label's.
        # This problem is not came from its original size, but its extension.
        if image.shape[:2] != label.shape:
            image = image.transpose(1,0,2)
        
        # If Image's ndim is not 3, then extend one dimension. 
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            
        # Data Augmentation
        if self.transform and 'Normalize' in [self.transform.__getitem__(i).__class__.__name__ for i in range(self.transform.__len__())]:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
            
            image = image.float()
            label = label.float()
        else:
            data = self.transform(image=image, mask=label)
            image = (data['image'] / 255).float()
            label = (data['mask']).float()
        
        data = {'name': name, 'input' : image, 'label' : label}
        
        return data


def get_dataloader(data_dir_: str, transform_: A.Compose, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Description
     : A function that makes a dataloader.

    Parameters
     : data_dir_ : a directory that includes image folder and label folder.
     : transform_ : albumentation compose.
     : batch_size : batch size

    Return
     : DataLoader 
    """
    return DataLoader(Custom_dataset(data_dir=data_dir_, transform=transform_), batch_size=batch_size, shuffle=shuffle, drop_last=True)