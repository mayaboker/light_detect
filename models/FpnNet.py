import torch.nn as nn
import torch
from common import ConvBn
from FpnHead import FpnHead

class FpnNet(nn.Module):
    def __init__(self, backbone, head_ch, reg_dict, upsample_mode='conv', one_feat_map=True):
        super(FpnNet, self).__init__()
        self.head_ch = head_ch
        self.reg_dict = reg_dict

        self.base = backbone
        num_laterals = len(backbone.feat_channels)
        self.fpn = FpnHead(head_ch, num_laterals, upsample_mode=upsample_mode, one_feat_map=one_feat_map)
        
        self.num_maps = 1 if one_feat_map else num_laterals
        self.reg_heads = [
            build_regression_head() for _ in range(self.num_maps)
        ]
       
            
    def build_regression_head(self):
        reg_head_dict = {}
        for head, num_channels in self.reg_dict.items():
            m_head = []
            m_head.append(
                nn.Conv2d(head_ch, num_channels, kernel_size=1, stride=1, bias=True)
            )
            if head == 'hm':
                m_head.append(
                    nn.Sigmoid()
                )
            reg_head_dict[head] = nn.Sequential(*m_head)

        return reg_head_dict
            

    def forward(self, x):
        feats = self.base(x)
        p_feats = self.fpn(feats)
        outs = []
        for i, p_feat in enumerate(p_feats):
            reg_outs = []
            for reg_head in self.reg_dict.keys():
                reg_outs.append(
                    self.reg_heads[i](p_feat)
                )
            outs.append(reg_outs)

        return outs


