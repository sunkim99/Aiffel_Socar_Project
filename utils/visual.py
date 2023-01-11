import os, cv2, glob, json, random, torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from pathlib import Path
from tqdm import tqdm
from copy import copy


def visual_img(img_path: str, mask_path: str, only_mask: bool=False):
    '''
    Description
     : A function to visualize a set of images. (i.e., image, mask, masked_image)

    Parameters
     : img_path : image path to visualize.
     : mask_path : mask path to visualize.
     : only_mask : if True, only visualize "masked image".

    Return
     : None
    '''
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    green_mask = mask.copy()
    green_mask[:,:,0] = 0
    green_mask[:,:,2] = 0

    masked_image = cv2.addWeighted(image, 1, green_mask, 0.5, 0)

    if only_mask:
        plt.imshow(masked_image)
        plt.axis('off')
        plt.title('Masked_Image')
        plt.show()
    else:
        fig, ax = plt.subplots(ncols=3, nrows=1)
        plt.subplots_adjust(wspace=0.1)

        for idx, (name, img) in enumerate({'Image' : image, 'Mask' : mask, 'Masked_Image' : masked_image}.items()):
            ax[idx].imshow(img)
            ax[idx].set_axis_off()
            ax[idx].set_title(name)
        plt.show()


def visual_mask(pred: torch.Tensor, target: torch.Tensor):
    '''
    Description
     : A function to visualize the prediction and compare with the target.
    
    Parameters
     : pred : segmentation model's output
     : target : mask image
    
    Return
     : None

    '''

    pred = (torch.sigmoid(pred) > 0.5).int().cpu().permute(1, 2, 0)
    target = target.permute(1, 2, 0)

    fig, ax = plt.subplots(ncols=2, nrows=1)
    plt.subplots_adjust(wspace=0.1)

    for idx, (name, img) in enumerate({'Pred' : pred, 'Target' : target}.items()):
        ax[idx].imshow(img)
        ax[idx].set_axis_off()
        ax[idx].set_title(name)
    plt.show()


def visual_dataset(image, label):
    image = ((image) * 0.5 + 0.5) * 255
    image = image.permute(1, 2, 0)
    label = label.permute(1, 2, 0) * 255
    
    
    image = image.numpy().astype('uint8')
    label = label.numpy().astype('uint8')
    label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
    masked = label.copy()
    masked[:,:,0] = 0
    masked[:,:,2] = 0
    
    masked = cv2.addWeighted(image, 1, masked, 1, 0)
    
    fig, ax = plt.subplots(ncols=3, nrows=1)
    plt.subplots_adjust(wspace=0.1)
    
    for i in range(3):
        ax[0].imshow(image)
        ax[1].imshow(masked)
        ax[2].imshow(label, cmap='gray')
        
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        ax[2].set_axis_off()
        
        ax[0].set_title('Image')
        ax[1].set_title('Masked')
        ax[2].set_title('Mask')
        
    plt.show()