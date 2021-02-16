import torch.nn as nn
import torch

class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, k, s):
        super(ConvBn, self).__init__()
        p = k //2
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class ConvTransposeBn(nn.Module):
    def __init__(self, in_c, out_c, k, s):
        super(ConvTransposeBn, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=0, output_padding=0, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class DwConvBn(nn.Module):
    def __init__(self, in_c, k, s=1):
        super(DwConvBn, self).__init__()
        p = k //2
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=k, stride=s, padding=p, groups=in_c),
            nn.BatchNorm2d(in_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)