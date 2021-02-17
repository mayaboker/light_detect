import os
import torch
import torch.utils.data
import torch.nn.functional as F
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import json


class CAVIARDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment=None):
        self.augment = augment 
        f_data = root + '_gt.json'
        with open(f_data, 'r') as f:
            data = json.load(f)
        self.f_imgs = []
        self.labels = []
        for i, k in enumerate(data.keys()):
            self.f_imgs.append(os.path.join(root, data[k]['img_name']))
            l = data[k]['bbox']
            self.labels.append([ll[:5] for ll in l])
                
    def __getitem__(self, index):
        img = Image.open(self.f_imgs[index])
        label = self.labels[index]
        if self.augment:
            pass
        return img, label

    def __len__(self):
        return len(self.f_imgs)


if __name__ == "__main__":
    root = '/home/core4/data/CAVIAR/Browse1'
    D = CAVIARDataset(root)
    img, labels = next(iter(D))
    img = np.array(img)
    import matplotlib.pyplot as plt
    import cv2
    for l in labels:
        cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()