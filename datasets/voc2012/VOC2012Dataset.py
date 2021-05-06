import os
import torch
import torch.utils.data
import numpy as np
import json
from pathlib import Path
from datasets.PedestrianDataset import PedestrianDataset
from PIL import Image
from utils.utils import load_yaml

class VOCDataset(PedestrianDataset):
    def __init__(self, root, augment, strides=(4,), mode='train'):        
        cfg = load_yaml('config.yaml')
        self.min_area = cfg['train']['transforms']['min_area']
        self.in_size = cfg['train']['transforms']['in_size']
        super().__init__(root, augment, strides, mode)

    def parse_data(self, root, mode):
        f_data = Path(root) / f'voc_gt_{mode}.json'        
        with open(f_data, 'r') as f:
            data = json.load(f)                    
        f_imgs = []
        labels = []
        for _, k in enumerate(data.keys()):
            has_boxes = self.is_resized_contain_boxes(data[k]['bbox'], os.path.join(root, 'JPEGImages', k))
            if has_boxes:
                f_imgs.append(os.path.join(root, 'JPEGImages', k))
                boxes = data[k]['bbox']
                for i, b in enumerate(boxes):
                    boxes[i] = [float(bb) for bb in b]                                        
                labels.append(boxes)
            else:
                print("HARYAN ARUR")

        return f_imgs, labels


    def name(self):
        return 'voc'

if __name__ == '__main__':
    root = r""
    
