import torch.nn as nn

class AnchorFreeLoss(nn.Module):
    def __init__(self, cfg):
        super(AnchorFreeLoss, self).__init__()
