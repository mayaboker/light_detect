from models import resnet
from models.BackLite import Backbone
from models.resnet import resnet50
from models.FpnNet import FpnNet
from datasets.caviar.CAVIARDataset import CAVIARDataset
from datasets.voc2012.VOC2012Dataset import VOCDataset
from datasets.virat.ViratDataset import ViratDataset
from datasets.MultiDataset import MultiDataset

from models.tf.BackLite import Backbone as BackboneTf
from models.tf.FpnNet import FpnNet as FpnNetTf

dataset_paths_dict = {
    'caviar':   'caviar_dir',
    'voc':      'voc_dir',
    'virat':    'virat_dir'      
}

def get_pedestrian_dataset(dataset_name, paths, augment, strides=(4,), mode='train', multi_datasets=None):
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
    elif dataset_name == 'virat':
        dataset = ViratDataset(
            paths['virat_dir'],
            augment,
            strides=strides,
            mode=mode
        )
    elif dataset_name == 'multi':
        datasets_list = []
        for data in multi_datasets:
            datasets_list.append(
                get_pedestrian_dataset(
                    data,
                    paths,
                    augment,
                    strides=strides,
                    mode=mode
                )
            )
        dataset = MultiDataset(datasets_list)
    else:
        raise Exception(f'Not implemented dataset: {dataset_name}')

    return dataset


def get_fpn_net(cfg_net, framework='torch'):
    head_ch = cfg_net['head_channels']
    reg_dict = cfg_net['channels_dict']
    one_feat_map = cfg_net['one_feat_map']
    upsample_mode = cfg_net['upsample']

    if cfg_net['backbone'] == 'lite':
        backbone = Backbone(head_ch)
    elif cfg_net['backbone'] == 'resnet':
        backbone = resnet50()
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
    paths = cfg['paths']
    train_cfg = cfg['train']
    dataset_name = train_cfg['dataset']
    datasets_multi = train_cfg['multi_datasets']

    D = get_pedestrian_dataset(
        dataset_name,
        paths,
        augment=get_train_transforms(train_cfg['transforms']),
        mode='train',
        multi_datasets=datasets_multi
    )

    for d, s in zip(D.datasets, D.seperators):
        print(f'Name: {d.name()}', len(d), s)

    for i in range(len(D)):
        data_ind, ind = D.get_actual_index(i)
        if ind >= len(D.datasets[data_ind])-1:
            print(f'#{i}:\t Dataset: {data_ind},\t index: {ind}')
        # img, hms = D[i]
    print()
    
