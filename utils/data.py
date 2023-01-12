import os, cv2, glob, json, random
import numpy as np

from collections import Counter
from pathlib import Path
from tqdm import tqdm


def get_datasize(root_dir: str, damage: str, type: str):
    '''
    Description
     : A function to get the number of dataset
    '''
    print(f"Image file size : {len(glob.glob(f'{root_dir}/{damage}/{type}/image/*'))} | Mask file size : {len(glob.glob(f'{root_dir}/{damage}/{type}/label/*'))}")


def make_maskfile(img: np.ndarray, ann_file: json):
    ''' 
    Description
     : A function to make mask images of an image seperately.

    Parameters
     : img : image
     : ann_file : annotation file
    '''
    damage_name = set([i['damage'] for i in ann_file['annotations'] if i['damage'] != None])
    damage_part = dict([[i, []] for i in damage_name])
    
    erase = [key for key in damage_part.keys() if key not in ['Scratched', 'Crushed', 'Separated']]
    
    for i in erase:
        del damage_part[i]
    
    put_seg = lambda x : damage_part[x['damage']].append([np.array(x['segmentation'])][0])
    
    for i in range(len(ann_file['annotations'])):
        if ann_file['annotations'][i]['damage'] in damage_part.keys():
            x = ann_file['annotations'][i]
            x = [np.squeeze(i) for i in ann_file['annotations'][i]['segmentation']][0]
            if len(x.shape) != 2:
                while len(x.shape) != 2:
                    x = [np.squeeze(i) for i in ann_file['annotations'][i]['segmentation']][0]
                    x = [np.squeeze(i) for i in x][0]
            ann_ = {'damage' : ann_file['annotations'][i]['damage'], 'segmentation' : x.tolist()}
            put_seg(ann_)
    try:
        mask_file = dict(map(lambda x : (x[0], cv2.fillPoly(np.zeros_like(img), x[1], color=(255, 255, 255))), damage_part.items()))
    except:
        print('[*] Something is wrong...')
    return mask_file


def get_annotation(root_dir: str, damage: str, type: str, name: str):
    """
    Description
     : A function to get image file and annotation file.
    
    Return
     : image
     : annotation file
    """
    imgpath = glob.glob(f'{root_dir}/{damage}/{type}/image/{name}*')
    for i in imgpath:
        image = cv2.imread(i)
    
    return image, json.load(open(glob.glob(f'{root_dir}/{damage}/{type}/label/{name}*')[0]))


def make_maskfile(img, ann_file):
    damage_name = set([i['damage'] for i in ann_file['annotations'] if i['damage'] != None])
    damage_part = dict([[i, []] for i in damage_name])
        
    erase = [key for key in damage_part.keys() if key not in ['Scratched', 'Crushed', 'Separated']]
    
    for i in erase:
        del damage_part[i]
    
    for key in damage_part.keys():
        if not key in ['Scratched', 'Crushed', 'Separated']:
            del damage_part[key]
    
    put_seg = lambda x : damage_part[x['damage']].append([np.squeeze(i) for i in x['segmentation']][0])
    
    for i in range(len(ann_file['annotations'])):
        if ann_file['annotations'][i]['damage'] in damage_part.keys():
            put_seg(ann_file['annotations'][i])
    mask_file = dict(map(lambda x : (x[0], cv2.fillPoly(np.zeros_like(img), x[1], color=(255, 255, 255))), damage_part.items()))
    return mask_file


def make_maskfile_2(img, ann_file):
    damage_name = set([i['damage'] for i in ann_file['annotations'] if i['damage'] != None])
    damage_part = dict([[i, []] for i in damage_name])
    
    erase = [key for key in damage_part.keys() if key not in ['Scratched', 'Crushed', 'Separated']]
    
    for i in erase:
        del damage_part[i]
    
    put_seg = lambda x : damage_part[x['damage']].append([np.array(x['segmentation'])][0])
    
    for i in range(len(ann_file['annotations'])):
        if ann_file['annotations'][i]['damage'] in damage_part.keys():
            x = ann_file['annotations'][i]
            x = [np.squeeze(i) for i in ann_file['annotations'][i]['segmentation']][0]
            if len(x.shape) != 2:
                while len(x.shape) != 2:
                    x = [np.squeeze(i) for i in ann_file['annotations'][i]['segmentation']][0]
                    x = [np.squeeze(i) for i in x][0]
            ann_ = {'damage' : ann_file['annotations'][i]['damage'], 'segmentation' : x.tolist()}
            put_seg(ann_)
    try:
        mask_file = dict(map(lambda x : (x[0], cv2.fillPoly(np.zeros_like(img), x[1], color=(255, 255, 255))), damage_part.items()))
    except:
        print('** Something is wrong...')
    return mask_file


def make_img_mask(root_dir: str, damage: str, type: str):
    
    name_list = [i.split('/')[-1].split('.')[0] for i in glob.glob(f'{root_dir}/{damage}/{type}/image/*')]
    img_dict = {}
    for name in name_list:
        img, ann_file = get_annotation(damage, name)
        try:
            mask_file = make_maskfile(img, ann_file)
        except:
            mask_file = make_maskfile_2(img, ann_file)
    
        img_dict[name] = {'image' : img, 'mask_file' : mask_file}
        
    count_dict = {'scratched' : 0, 'separated' : 0, 'crushed' : 0}
    
    for i in [i['mask_file'] for i in img_dict.values()]:
        for name in i:
            count_dict[name.lower()] += 1
    
    return img_dict, count_dict


def make_directory(save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)

    crushed_path = save_path / 'crushed'
    scratched_path = save_path / 'scratched'
    separated_path = save_path / 'separated'
    
    data_path = {}
    
    for path in [crushed_path, scratched_path, separated_path]:
        img_path = path / 'image'
        img_path.mkdir(parents=True, exist_ok=True)

        label_path = path / 'label'
        label_path.mkdir(parents=True, exist_ok=True)
            
        data_path[str(path).split('/')[1]] = {'image' : img_path, 'label' : label_path}
    
    return data_path


def socar_data_expand(damage_name, save_path):
    img_dict, count_dict = make_img_mask(damage_name)
    data_path = make_directory(save_path)
    print(f'[*] {damage_name.capitalize()} data Expansion is processing...')
    print(f'Before expansion dataset : ')
    get_datasize(damage_name)
    for batch_idx, (image, mask) in tqdm(enumerate(img_dict.items())):
        for key, val in mask['mask_file'].items():
            cv2.imwrite(str(data_path[key.lower()]['label'] / f'{image}.jpg'), val)
            cv2.imwrite(str(data_path[key.lower()]['image'] / f'{image}.jpg'), img_dict[image]['image'])
    print(f'{count_dict} is expanded')