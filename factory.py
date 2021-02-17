from models.BackLite import Backbone
from models.FpnNet import FpnNet


def get_fpn_net(cfg_net):
    head_ch = cfg_net['head_channels']
    reg_dict = cfg_net['channels_dict']
    one_feat_map = cfg_net['one_feat_map']

    backbone = Backbone(head_ch)
    return FpnNet(backbone, head_ch, reg_dict, one_feat_map=one_feat_map)
