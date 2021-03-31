import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO - replace resize to keep aspect ratio
def get_train_transforms(cfg_trans):
    min_area = cfg_trans['min_box'] * cfg_trans['min_box']
    trans_list = A.Compose([        
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.HorizontalFlip(p=0.5),                
        A.RandomResizedCrop(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=0.5), 
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        A.Normalize(mean=3*[0], std=3*[1]),        
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=0.4)
    )
    return trans_list


def get_val_transforms(cfg_trans):
    min_area = cfg_trans['min_box'] * cfg_trans['min_box']
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),        
        A.Normalize(mean=3*[0], std=3*[1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=0.4)
    )
    return trans_list


def get_test_transforms(cfg_trans):
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.Normalize(mean=3*[0], std=3*[1]),        
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc')
    )
    return trans_list


if __name__ == "__main__":
    root = '/home/core4/data/CAVIAR/val'
    gt = '/home/core4/data/CAVIAR/val/val_gt.json'
    cfg_trans = {}
    cfg_trans['min_box'] = 10
    cfg_trans['in_size'] = [320, 320]
    transforms = get_train_transforms(cfg_trans)
    import cv2, json
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image
    img = np.array(Image.open(root + '/Rest_InChair/000000.png'))
    with open(gt, 'r') as f:
        data = json.load(f)
    labels = data['Rest_InChair/000000.png']['bbox']
    transformed = transforms(image=img, bboxes=labels)
    img = transformed['image']
    img = ((np.array(to_pil_image(img)) / 2 + 0.5) * 255).astype(np.uint8)
    labels = transformed['bboxes']
    for l in labels:
        cv2.rectangle(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 1)
    plt.imshow(img)
    plt.show()
    