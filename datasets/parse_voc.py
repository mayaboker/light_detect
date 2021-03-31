import torch
from pathlib import Path
import pandas as pd
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from api import decode
from torchvision.transforms.functional import to_pil_image

classes = dict(zip(['person', 'car', 'bus'], range(3)))

def preprocess_voc2012(path, categories):
    annotations_path = path / 'Annotations'
    all_annotations = annotations_path.glob('**/*.xml')
    filenames = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    objects_category = []
    for annotation_path in all_annotations:
        annotation_path_str = str(annotation_path)
        curr_annotation = minidom.parse(annotation_path_str)
        filename = curr_annotation.getElementsByTagName('filename').item(0).firstChild.data
        objects = curr_annotation.getElementsByTagName('object')
        for curr_object in objects:
            category = curr_object.getElementsByTagName('name').item(0).firstChild.data
            if category in categories:
                bounding_box = curr_object.getElementsByTagName('bndbox')
                xmin = float(bounding_box[0].getElementsByTagName('xmin').item(0).firstChild.data)
                ymin = float(bounding_box[0].getElementsByTagName('ymin').item(0).firstChild.data)
                xmax = float(bounding_box[0].getElementsByTagName('xmax').item(0).firstChild.data)
                ymax = float(bounding_box[0].getElementsByTagName('ymax').item(0).firstChild.data)
                filenames.append(filename)
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                objects_category.append(category)


    ii = np.random.permutation(len(filenames)).astype(np.int)
    N_TRAIN = int(len(filenames) * 0.7)
    N_VAL = int(len(filenames) * 0.2)        
    ii_train = ii[:N_TRAIN]
    ii_val = ii[N_TRAIN:N_TRAIN+N_VAL]
    ii_test = ii[N_TRAIN+N_VAL:]

    for i in ['train', 'val', 'test']:
        if i == 'train':
            data_index = ii_train
        if i == 'val':    
            data_index = ii_val 
        if i == 'test':
            data_index = ii_test
        
        filenames_ = [filenames[j] for j in data_index]
        category_ = [classes[objects_category[j]] for j in data_index]
        xmins_ = [xmins[j] for j in data_index]
        ymins_ = [ymins[j] for j in data_index]
        xmaxs_ = [xmaxs[j] for j in data_index]
        ymaxs_ = [ymaxs[j] for j in data_index]
            
        data = {
            'filename': filenames_,
            'category': category_,
            'xmin': xmins_,
            'ymin': ymins_,
            'xmax': xmaxs_,
            'ymax': ymaxs_
        }
        df = pd.DataFrame(data=data, columns=['filename', 'category', 'xmin', 'ymin', 'xmax', 'ymax'])

        print(df.head())

        df.to_csv(path / f'voc_data_{i}.csv')
        

if __name__ == '__main__':
    voc_root = '/home/core4/data/pascal_voc_seg/VOCdevkit/VOC2012'
    path_to_annotations = Path(voc_root)
    categories = ['person', 'bus', 'car']
    preprocess_voc2012(path_to_annotations, categories)
# class VOC2012Dataset(torch.utils.data.Dataset):
#     pass