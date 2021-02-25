import math
import os
import torch
import torch.utils.data
import torch.nn.functional as F
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import json
from utils.image_utils import gaussian_radius, draw_umich_gaussian


class CAVIARDataset(torch.utils.data.Dataset):
    def __init__(self, root, augment=None, strides=(4,)):
        # TODO - get min box from configs
        self.min_box = 8
        self.strides = strides
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
        img = self.load_image(self.f_imgs[index])       
        # img = Image.open(self.f_imgs[index])
        labels = self.labels[index]
        if self.augment:
            transformed = self.augment(image=img, bboxes=labels)
            img = transformed['image']
            labels = transformed['bboxes']
        
        heatmaps = []
        for stride in self.strides:
            heatmaps.append(self.make_heatmaps(img, labels, stride))
        return img, heatmaps

    def __len__(self):
        return len(self.f_imgs)

    def load_image(self, path):
        img = Image.open(path)
        return np.array(img, dtype=np.float)


    def make_heatmaps(self, im, bboxes, stride):
        height, width = im.size()[1:]
        out_width = int(width / stride)
        out_height = int(height / stride)

        res = np.zeros([5, out_height, out_width], dtype=np.float16)
        for bbox in bboxes:
            #0 heatmap
            left, top, right, bottom = map(lambda x: x / stride, bbox[:4])
            if right <= left or bottom <= top:
                continue
            if bbox[0] + self.min_box >= bbox[2] or bbox[1] + self.min_box >= bbox[3]:
                continue
            x = int((left + right) /2)
            y = int((top + bottom) /2)

            x, y = min(max(x, 0), out_width-1), min(max(y, 0), out_height-1)
            h, w = (bottom - top), (right - left)

            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(int(radius), 0)
            draw_umich_gaussian(res[0], (x, y), radius)

            # 1, 2 center offset
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            res[1][y, x] = center_x - x
            res[2][y, x] = center_y - y

            # 3, 4 width and height
            eps = 1e-6
            res[3][y, x] = np.log(w + eps)
            res[4][y, x] = np.log(h + eps)

        return res


if __name__ == "__main__":
    from transformations import get_train_transforms
    from utils.utils import load_yaml
    cfg = load_yaml('config.yaml')
    root = cfg['paths']['data_dir']
    D = CAVIARDataset(root, augment=get_train_transforms(cfg['train']['transforms']))

    img, hms = D[20]
    
    import matplotlib.pyplot as plt
    import cv2
    show = np.zeros([hms[0].shape[1], hms[0].shape[2], 3])
    show[:, :, 0] = hms[0][0]
    show = cv2.resize(show, (640, 640))
    img = img.permute(1, 2, 0).numpy()
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    show = img + show

    plt.imshow(show, cmap='hot', interpolation='nearest')
    plt.savefig('heatmap.png')
    
    # while True:
    #     img, labels = next(iter(D))
    #     img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    #     import cv2
    #     for l in labels:
    #         cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 1)
    #     cv2.imshow('', img)
    #     k = cv2.waitKey(0)
    #     if k==27:    # Esc key to stop
    #         break        
    #     else:
    #         cv2.destroyAllWindows()            