import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(cfg_trans):
    min_area = cfg_trans['min_area']
    trans_list = A.Compose([        
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.HorizontalFlip(p=0.5),                
        A.RandomResizedCrop(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=0.5), 
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        A.Normalize(mean=3*[0], std=3*[1]),        
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=0.4)
    )
    return trans_list


def get_val_transforms(cfg_trans):
    min_area = cfg_trans['min_area']
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),        
        A.Normalize(mean=3*[0], std=3*[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=0.4)
    )
    return trans_list


def get_test_transforms(cfg_trans):
    min_area = cfg_trans['min_area']
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.Normalize(mean=3*[0], std=3*[1]),        
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=0.4)
    )
    return trans_list


if __name__ == "__main__":
    root = '/home/core4/data/CAVIAR/val'
    gt = '/home/core4/data/virat/frames/VIRAT_S_000200_00_000100_000171/labels_gt.json'
    cfg_trans = {}
    cfg_trans['min_area'] = 600
    cfg_trans['in_size'] = [1280, 720]
    transforms = get_test_transforms(cfg_trans)
    import cv2, json
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image
    #img = np.array(Image.open(root + '/Rest_InChair/000000.png'))
    img = np.array(Image.open('/home/core4/data/virat/frames/VIRAT_S_000200_00_000100_000171/000645.png'))
    with open(gt, 'r') as f:
       data = json.load(f)
    labels = data['VIRAT_S_000200_00_000100_000171/000645.png']
    labels[0] = np.array(labels[0]).astype(np.float)
    transformed = transforms(image=img, bboxes=labels)
    img = transformed['image']
    img = np.array(to_pil_image(img))
    labels = transformed['bboxes']    
    for l in labels:
        print((l[2]-l[0])*(l[3]-l[1]))
        cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()
    