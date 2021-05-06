import torch
from pathlib import Path
import pandas as pd
from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import cv2
from api import decode
from torchvision.transforms.functional import to_pil_image
import xmltodict, json
from PIL import Image

classes = dict(zip(['person', 'car', 'bus'], range(3)))

DEBUG = False

def preprocess_voc2012(path, categories):
    annotations_path = path / 'Annotations'
    all_annotations = annotations_path.glob('**/*.xml')        
    gt = dict()
    img_names = []
    for annotation_path in all_annotations:
        annotation_path_str = str(annotation_path)
        with open(annotation_path_str.__str__()) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())
        xml_file.close()            
        data = data_dict['annotation']
        img_name = data['filename']
        obj = data['object']
        bbox = []
        if isinstance(obj, list):
            for obj_k in obj:
                if 'actions' in obj_k.keys():
                    continue
                cat = obj_k['name']
                if cat in categories:
                    bbox_str = obj_k['bndbox']
                    x_min = float(bbox_str['xmin'])
                    x_max = float(bbox_str['xmax'])
                    y_min = float(bbox_str['ymin'])
                    y_max = float(bbox_str['ymax'])  
                    cat = 1#classes[cat]              
                    bbox.append([x_min, y_min, x_max, y_max, cat])                
        else:
            if 'actions' in obj.keys():
                continue
            cat = obj['name']
            if cat in categories:
                bbox_str = obj['bndbox']
                x_min = float(bbox_str['xmin'])
                x_max = float(bbox_str['xmax'])
                y_min = float(bbox_str['ymin'])
                y_max = float(bbox_str['ymax'])
                cat = categories.index(cat)
                bbox.append([x_min, y_min, x_max, y_max, cat])                            
        if len(bbox) > 0:
            gt[img_name] = {'bbox': bbox, 'img_name': img_name}
            img_names.append(img_name)
            if DEBUG:
                img = np.array(Image.open('/home/core4/data/pascal_voc_seg/VOCdevkit/VOC2012/JPEGImages/'+img_name))            
                bbox = gt[img_name]['bbox']
                for i, l in enumerate(bbox):
                    img = cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 1)
                plt.imshow(img)
                plt.show()
    
    n_total = len(gt)
    n_train = int(0.7 * n_total)
    n_val = int(0.2 * n_total)
    ii = np.random.permutation(n_total)
    ii_train = ii[:n_train]
    ii_val = ii[n_train:n_val+n_train]
    ii_test = ii[n_train+n_val:]
    print('train:{} val:{} test:{}'.format(ii_train.shape[0], ii_val.shape[0], ii_test.shape[0]))
    gt_train = dict()
    for i in ii_train:
        im = img_names[i]
        gt_train[im] = gt[im]

    with open('/home/core4/data/pascal_voc_seg/VOCdevkit/VOC2012/voc_gt_train.json', "w") as write_file:
        json.dump(gt_train, write_file)

    gt_val = dict()
    for i in ii_val:
        im = img_names[i]
        gt_val[im] = gt[im]

    with open('/home/core4/data/pascal_voc_seg/VOCdevkit/VOC2012/voc_gt_val.json', "w") as write_file:
        json.dump(gt_val, write_file)

    gt_test = dict()
    for i in ii_test:
        im = img_names[i]
        gt_test[im] = gt[im]        

    with open('/home/core4/data/pascal_voc_seg/VOCdevkit/VOC2012/voc_gt_test.json', "w") as write_file:
        json.dump(gt_test, write_file)

if __name__ == '__main__':
    voc_root = '/home/core4/data/pascal_voc_seg/VOCdevkit/VOC2012'
    path_to_annotations = Path(voc_root)
    categories = ['person']
    preprocess_voc2012(path_to_annotations, categories)
