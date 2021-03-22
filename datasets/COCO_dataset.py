from torch.utils.data import Dataset
from PIL import Image
import json
import glob
import re
import numpy as np
from utils.image_utils import gaussian_radius, draw_umich_gaussian
import math
import cv2
import time


'''
class COCODataset(Dataset):
    def __init__(self, data_path, augment=None, strides=(4,)):
        file_data = data_path + '/instances_val2017.json'
        self.labels = []
        self.strides = strides
        self.f_imgs = glob.glob(data_path + '/truck_images/' + '*.jpg')
        with open(file_data, 'r') as f:
            data_instances = json.load(f)
            for img in self.f_imgs:
                img_name = int(re.search(r"\d+[.jpg]", img).group()[:-1])
                specific_id_sample = []
                image_ids = []
                count = 0
                for sample in data_instances["annotations"]:
                    if sample["image_id"] == img_name:
                        specific_id_sample = sample["bbox"]
                        specific_id_sample[2] = specific_id_sample[0] + specific_id_sample[2]
                        specific_id_sample[3] = specific_id_sample[1] + specific_id_sample[3]
                        specific_id_sample += [sample["category_id"]]
                        #specific_id_sample[0] = sample["bbox"][0]
                        #specific_id_sample[1] = sample["bbox"][1]
                        #specific_id_sample[2] = sample["bbox"][0] + sample["bbox"][2]
                        #specific_id_sample[3] = sample["bbox"][1] + sample["bbox"][3]
                        count += 1
                        image_ids.append(specific_id_sample)                                                
                self.labels.append(specific_id_sample)
        self.augment = augment
    '''

class COCODataset(Dataset):
    def __init__(self, data_path, augment=None, strides=(4,)):
        file_data = data_path + '/instances_val2017.json'
        self.labels = []
        self.strides = strides
        self.f_imgs = glob.glob(data_path + '/truck_images/' + '*.jpg')
        self.min_box = 8
        with open(file_data, 'r') as f:
            data_instances = json.load(f)
            for i, img in enumerate(self.f_imgs):
                img_name = int(re.search(r"\d+[.jpg]", img).group()[:-1])
                specific_id_sample = []
                image_ids = []
                count = 0
                for sample in data_instances["annotations"]:
                    if sample["image_id"] == img_name:
                        label = sample["bbox"]
                        label[2] = label[0] + label[2]
                        label[3] = label[1] + label[3]
                        label += [sample["category_id"]]
                        specific_id_sample.append(label)                                               
                image_ids.append(specific_id_sample)                                                
                self.labels.append(specific_id_sample)
        self.augment = augment



    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        labels = self.labels[index]
        img = self.load_image(self.f_imgs[index])
        if self.augment:
            transformed = self.augment(image=img, bboxes=labels)
            img = transformed['image']
            labels = transformed['bboxes']

        heatmaps = []
        for stride in self.strides:
            heatmaps.append(self.make_heatmaps(img, labels, stride))
        return img, heatmaps
    
    def load_image(self, path):
        img = Image.open(path)        
        # TODO : taking care of no channels situation
        img = np.array(img)
        if len(img.shape) < 3:            
            start_time = time.time()
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)            
           # cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #     img_expand = np.expand_dims(img, axis=0)
        #     img_expand[0] = img
        #     img_expand[1] = img
        #     img_expand[2] = im

        return img
    
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


if __name__ == '__main__':
    from transformations import get_train_transforms
    from utils.utils import load_yaml
    cfg = load_yaml('config.yaml')
    annotations_file = r'/home/core2/Documents/truck/annotations'
    coco = COCODataset(annotations_file, augment=get_train_transforms(cfg['train']['transforms']), strides=(4, 8))
    img, heatmaps = coco.__getitem__(3)
    print()
