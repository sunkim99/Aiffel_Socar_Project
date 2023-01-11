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


class Custom_dataset(torch.utils.data.Dataset):
    """
    Description
     : A class to make a customized dataset.

    Parameters
     : data_dir : a directory that includes image folder and label folder.
     : transform : albumentation compose.

    """
    def __init__(self, data_dir: str, transform: A.Compose = None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.image_dir = self.data_dir + 'image'
        self.label_dir = self.data_dir + 'label'
        
        lst_image = os.listdir(self.image_dir)
        lst_label = os.listdir(self.label_dir)
        
        self.lst_label = lst_label
        self.lst_image = lst_image
        
    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index: int):
        name = self.lst_image[index].split('.')[0]
        image = np.array(Image.open(glob.glob(self.image_dir + f'/{name}*')[0])).astype('uint8')
        label = Image.open(glob.glob(self.label_dir + f'/{name}*')[0])
        label = (np.array(label.convert('L')) != 0).astype('uint8')

        if image.shape[:2] != label.shape:
            image = image.transpose(1,0,2)
        
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            
        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']
            
            image = image.float()
            label = label.float()
#             data_img /= 255
            
        data = {'name': name, 'input' : image, 'label' : label}
        
        return data


def get_dataloader(data_dir_: str, transform_: A.Compose, batch_size: int) -> DataLoader:
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
    return DataLoader(Custom_dataset(data_dir=data_dir_, transform=transform_), batch_size=batch_size, shuffle=True, drop_last=True)