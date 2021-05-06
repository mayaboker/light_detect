import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm

def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))

def WNConvTranspose2d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose2d(*args, **kwargs))

class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, k, s):
        super(ConvBn, self).__init__()
        p = k //2
        self.conv = nn.Sequential(
            WNConv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),            
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ConvTransposeBn(nn.Module):
    def __init__(self, in_c, out_c, k, s):
        super(ConvTransposeBn, self).__init__()
        self.conv = nn.Sequential(            
            WNConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=0, output_padding=0, bias=False),            
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class DwConvBn(nn.Module):
    def __init__(self, in_c, k, s=1):
        super(DwConvBn, self).__init__()
        p = k //2
        self.conv = nn.Sequential(
            #nn.ReflectionPad2d(p),
            WNConv2d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c, bias=False),            
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MaxPoolNms(nn.Module):
    def __init__(self, kernel=3):
        super().__init__()
        self.kernel = kernel
        self.padding = (kernel -1) // 2
    
    def forward(self, x):
        hmax = F.max_pool2d(x, self.kernel, stride=1, padding=self.padding)
        keep = torch.floor(x - hmax + 1)
        return x * keep

