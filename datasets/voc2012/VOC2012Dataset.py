import os
import torch
import torch.utils.data
import numpy as np
import json
from pathlib import Path
from datasets.PedestrianDataset import PedestrianDataset

class VOCDataset(PedestrianDataset):
    def __init__(self, root, augment, strides=(4,), mode='train'):
        super().__init__(root, augment, strides, mode)

    def parse_data(self, root, mode):
        f_data = Path(root) / f'voc_gt_{mode}.json'        
        with open(f_data, 'r') as f:
            data = json.load(f)                    
        f_imgs = []
        labels = []

        for _, k in enumerate(data.keys()):
            f_imgs.append(os.path.join(root, 'JPEGImages', data[k]['img_name']))
            l = data[k]['bbox']
            labels.append([ll[:5] for ll in l])

        return f_imgs, labels

    def name(self):
        return 'voc'
    
