import torch.nn as nn
import torch
from torch.nn.modules.padding import ReflectionPad1d
from models.common import ConvBn
from models.FpnHead import FpnHead
from models.BackLite import Backbone
from torch.nn.utils import weight_norm

def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))

class PNet(nn.Module):
    def __init__(self, nf=32):
        super(PNet, self).__init__()
        model = [
            nn.ReflectionPad2d(1),
            WNConv2d(3, nf, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),

            nn.ReflectionPad2d(2),
            WNConv2d(nf, nf, kernel_size=3, stride=1, padding=0, dilation=2),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            WNConv2d(nf, nf, kernel_size=3, stride=1, padding=0, dilation=3),
            nn.ReLU(True),
        ]
        self.model_convstack = nn.Sequential(*model)
        
        model = [
            WNConv2d(nf, nf*2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            
            nn.ReflectionPad2d(1),
            WNConv2d(nf*2, nf*2, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),

            WNConv2d(nf*2, nf*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            
            nn.ReflectionPad2d(1),
            WNConv2d(nf*4, nf*4, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
        ]
        self.down = nn.Sequential(*model)
        nf *= 4

        model = [
            WNConv2d(nf, nf, 1, 1)
        ]
        self.bottleneck = nn.Sequential(*model)

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
        return  


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
