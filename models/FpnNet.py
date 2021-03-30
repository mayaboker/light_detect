import torch.nn as nn
import torch
from models.common import ConvBn
from models.FpnHead import FpnHead
from models.BackLite import Backbone

class FpnNet(nn.Module):
    def __init__(self, backbone, head_ch, reg_dict, upsample_mode='interpolate', one_feat_map=True):
        super(FpnNet, self).__init__()
        self.head_ch = head_ch
        self.reg_dict = reg_dict

        self.base = backbone
        num_laterals = len(backbone.feat_channels)
        self.fpn = FpnHead(head_ch, num_laterals, upsample_mode=upsample_mode, one_feat_map=one_feat_map)
        
        self.num_maps = 1 if one_feat_map else num_laterals
        self.reg_heads = nn.ModuleList([
            self.build_regression_head() for _ in range(self.num_maps)
        ])
       
            
    def build_regression_head(self):
        reg_head_dict = nn.ModuleDict()
        for head, num_channels in self.reg_dict.items():
            m_head = []
            m_head.append(
                nn.Conv2d(self.head_ch, num_channels, kernel_size=1, stride=1, bias=True)
            )
            if head == 'hm':
                m_head.append(
                    nn.Sigmoid()
                )
            reg_head_dict[head] = nn.Sequential(*m_head)

        return reg_head_dict
            

    def forward(self, x, ret_feats=False):
        feats = self.base(x)
        p_feats = self.fpn(feats)
        outs = []
        for i, p_feat in enumerate(p_feats):
            reg_outs = []
            for reg_head in self.reg_dict.keys():
                reg_outs.append(
                    self.reg_heads[i][reg_head](p_feat)
                )
            outs.append(reg_outs)

        if ret_feats:
            return outs, feats
        return outs


if __name__ == "__main__":
    head_ch = 32
    reg_dict = {'hm': 1, 'of': 2, 'wh': 2}
    backbone = Backbone(head_ch)#Backbone(head_ch)
    net = FpnNet(backbone, head_ch, reg_dict, one_feat_map=False)
    x = torch.randn(1, 3, 640, 480)
    
    y = net(x)
    for e in y:
        print('==================')
        for j in e:
            print(j.shape)
