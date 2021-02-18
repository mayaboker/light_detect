import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize

def get_train_transforms(cfg_trans=None):
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()
    ])