import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO - replace resize to keep aspect ratio
def get_train_transforms(cfg_trans):
    min_area = cfg_trans['min_box'] * cfg_trans['min_box']
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0., 0., 0.], std=[1, 1, 1]),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=0.4)
    )
    return trans_list


def get_val_transforms(cfg_trans):
    min_area = cfg_trans['min_box'] * cfg_trans['min_box']
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=min_area, min_visibility=0.4)
    )
    return trans_list


def get_test_transforms(cfg_trans):
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.]),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc')
    )
    return trans_list