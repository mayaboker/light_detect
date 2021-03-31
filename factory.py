from models.BackLite import Backbone
from models.FpnNet import FpnNet

from models.tf.BackLite import Backbone as BackboneTf
from models.tf.FpnNet import FpnNet as FpnNetTf


def get_fpn_net(cfg_net, framework='torch'):
    head_ch = cfg_net['head_channels']
    reg_dict = cfg_net['channels_dict']
    one_feat_map = cfg_net['one_feat_map']
    upsample_mode = cfg_net['upsample']

    if framework == 'torch':
        backbone = Backbone(head_ch)
        return FpnNet(backbone, head_ch, reg_dict, one_feat_map=one_feat_map, upsample_mode=upsample_mode)
    else:
        backbone = BackboneTf(head_ch)
        return FpnNetTf(backbone, head_ch, reg_dict, one_feat_map=one_feat_map, upsample_mode=upsample_mode)
