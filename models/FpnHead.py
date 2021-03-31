import torch.nn as nn
import torch
from models.common import ConvBn, ConvTransposeBn


class UpSample(nn.Module):
    def __init__(self, in_c, mode='interpolate'):
        super(UpSample, self).__init__()
        
        if mode == 'conv':
            self.up = ConvTransposeBn(in_c, in_c, k=2, s=2)
        elif mode == 'interpolate':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvBn(in_c, in_c, k=1, s=1)
            )
        
    def forward(self, x):
        return self.up(x)


class FpnHead(nn.Module):
    def __init__(self, head_ch, num_laterals, upsample_mode='conv', one_feat_map=True):
        super(FpnHead, self).__init__()
        
        self.ups = nn.ModuleList([
            UpSample(head_ch, mode=upsample_mode)
            for _ in range(num_laterals-1)
        ])

        self.one_feat_map = one_feat_map
        self.num_outputs = 1 if one_feat_map else num_laterals
        self.conv_feats = nn.ModuleList([
            ConvBn(head_ch, head_ch, k=3, s=1)
            for _ in range(self.num_outputs)
        ])

    def forward(self, feats):
        x = feats[0]
        outs = []
        for i in range(1, len(feats)):
            if not self.one_feat_map:
                outs.append(
                    self.conv_feats[i-1](x))
            x = self.ups[i-1](x) + feats[i]
        outs.append(
            self.conv_feats[-1](x))
        return outs
