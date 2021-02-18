import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO - replace resize to keep aspect ratio
def get_train_transforms(cfg_trans):
    trans_list = A.Compose([
        A.Resize(height=cfg_trans['in_size'][1], width=cfg_trans['in_size'][0], p=1),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        # A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_area=10, min_visibility=0.2)
    )
    return trans_list