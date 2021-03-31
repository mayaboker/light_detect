import os
import torch
import torch.utils.data
import numpy as np
import json
from datasets.PedestrianDataset import PedestrianDataset

class ViratDataset(PedestrianDataset):
    def __init__(self, root, augment, strides=(4,), mode='train'):
        super().__init__(root, augment, strides, mode)

    def parse_data(self, root, mode):
        if mode == 'train':
            f_data = root + '/' + 'train_gt.json'
        elif mode == 'val':
            f_data = root + '/' + 'val_gt.json'
        elif mode == 'test':
            f_data = root + '/' + 'test_gt.json'

        with open(f_data, 'r') as f:
            data = json.load(f)

        f_imgs = []
        labels = []
        for _, k in enumerate(data.keys()):
            f_imgs.append(os.path.join(root, k))
            labels.append(data[k])

        return f_imgs, labels

if __name__ == "__main__":    
    from transformations import get_train_transforms
    from utils.utils import load_yaml
    cfg = load_yaml('config.yaml')
    root = cfg['paths']['virat_dir']
    D = ViratDataset(
            root,
            augment=get_train_transforms(cfg['train']['transforms']),
            mode='train'
        )
    for d in range(30, 100):
        img, hms, labels = D[d]

        import matplotlib.pyplot as plt
        import cv2
        from api import decode
        from torchvision.transforms.functional import to_pil_image

        show = np.zeros([hms[0].shape[1], hms[0].shape[2], 3])
        show[:, :, 0] = hms[0][0]
        show = np.array(to_pil_image(img))
        hm = torch.Tensor(hms[0][0]).unsqueeze(0).unsqueeze(0).float()
        of = torch.Tensor(hms[0][1:3]).unsqueeze(0).float()
        wh = torch.Tensor(hms[0][3:5]).unsqueeze(0).float()
        out = [[hm, of, wh]]
        boxes, scores = decode(out, [4], 0.15, K=100)
        plt.imshow(show, cmap='hot', interpolation='nearest')
        for i, l in enumerate(boxes[0]):    
            show = cv2.rectangle(show, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 2)    
        for i, l in enumerate(labels):
            show = cv2.rectangle(show, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 1)    
        plt.imshow(show)
        plt.show()
