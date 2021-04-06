from abc import ABC
from posix import listdir
import torch
import torch.utils.data
import numpy as np
from PIL import Image
from datasets.dataset_utils import make_heatmaps
from datasets.dataset_utils import padded_resize

class PedestrianDataset(torch.utils.data.Dataset, ABC):
    def __init__(self, root, augment, strides=(4,), mode='train'):
        self.strides = strides
        self.augment = augment
        self.mode = mode
        
        self.f_imgs, self.labels = self.parse_data(root, mode)

    def parse_data(self, root, mode):
        raise NotImplementedError('Abstract method')

    def name(self):
        raise NotImplementedError('Abstract method')
    
    def __getitem__(self, index): 
        img = self.load_image(self.f_imgs[index])       
        labels = self.labels[index]
        img, labels = padded_resize(img, (320, 320), labels)
        if self.mode == 'test':
            transformed = self.augment(image=img, bboxes=labels)
            img = transformed['image']
            labels = transformed['bboxes']       
            return img, np.array(labels, dtype=np.float32)
            
        else:            
            transformed = self.augment(image=img, bboxes=labels)            
            img = transformed['image']
            labels = transformed['bboxes']            

            heatmaps = []
            for stride in self.strides:
                heatmaps.append(make_heatmaps(img, labels, stride))
            return img, heatmaps
            

    def __len__(self):
        return len(self.f_imgs)

    def load_image(self, path):
        img = Image.open(path)
        return np.array(img, dtype=np.float32)

