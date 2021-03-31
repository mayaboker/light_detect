from models.BackLite import Backbone
from models.FpnNet import FpnNet
# from resnet import resnet50
from datasets.CAVIARDataset import CAVIARDataset
from datasets.VOC2012Dataset import VOCDataset

from models.tf.BackLite import Backbone as BackboneTf
from models.tf.FpnNet import FpnNet as FpnNetTf

def get_pedestrian_dataset(dataset_name, paths, augment, strides=(4,), mode='train'):
    if dataset_name == 'caviar':
        dataset = CAVIARDataset(
            paths['caviar_dir'],
            augment,
            strides=strides,
            mode=mode
        )
    elif dataset_name == 'voc':
        dataset = VOCDataset(
            paths['voc_dir'],
            augment,
            strides=strides,
            mode=mode
        )
    else:
        raise Exception(f'Not implemented dataset: {dataset_name}')

    return dataset


def get_fpn_net(cfg_net, framework='torch'):
    head_ch = cfg_net['head_channels']
    reg_dict = cfg_net['channels_dict']
    one_feat_map = cfg_net['one_feat_map']
    upsample_mode = cfg_net['upsample']

    backbone = Backbone(head_ch)
    #backbone = resnet50()
    if framework == 'torch':
        backbone = Backbone(head_ch)
        return FpnNet(backbone, head_ch, reg_dict, one_feat_map=one_feat_map, upsample_mode=upsample_mode)
    else:
        backbone = BackboneTf(head_ch)
        return FpnNetTf(backbone, head_ch, reg_dict, one_feat_map=one_feat_map, upsample_mode=upsample_mode)


if __name__ == "__main__":
    from transformations import get_train_transforms
    from utils.utils import load_yaml
    cfg = load_yaml('config.yaml')
    root = cfg['paths']['caviar_dir']
    D = CAVIARDataset(root, augment=get_train_transforms(cfg['train']['transforms']))
    
    img, hms = next(iter(D))
    print()
    
