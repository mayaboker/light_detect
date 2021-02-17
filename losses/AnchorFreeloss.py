import torch.nn as nn
import torch
from losses.loss import ASLoss, FocalLoss, RegL1Loss

class AnchorFreeLoss(nn.Module):
    def __init__(self, cfg_train):
        super(AnchorFreeLoss, self).__init__()

        # hm loss
        hm_loss = cfg_train['losses']['hm']
        if hm_loss == 'asl':
            self.heatmap_loss = ASLoss(
                gamma_p=cfg_train['asl_params']['gamma_p'],
                gamma_n=cfg_train['asl_params']['gamma_n'],
                margin=cfg_train['asl_params']['margin']
            )
        elif hm_loss == 'focal':
            self.heatmap_loss = FocalLoss()
        else:
            raise Exception(f'Hm loss not valid: {hm_loss}')

        # wh loss
        wh_loss = cfg_train['losses']['wh']
        if wh_loss == 'sl1':
            self.scale_loss = RegL1Loss()
        else:
            raise Exception(f'Wh loss not valid: {wh_loss}')
        
        # of loss
        of_loss = cfg_train['losses']['of']
        if of_loss == 'sl1':
            self.offset_loss = RegL1Loss()
        else:
            raise Exception(f'Of loss not valid: {of_loss}')

    def forward(self, out, labels):
        losses = {
            'hm': 0,
            'wh': 0,
            'of': 0
        }
        probes = torch.zeros(2)
        num_maps = len(out)
        for stride_out, stride_labels in zip(out, labels):
            hm_loss, ps = self.heatmap_loss(stride_out[0], stride_labels[0])
            losses['hm'] += hm_loss
            probes += ps 

            pos_mask = stride_labels[0].eq(1)
            pos_mask = pos_mask.expand_as(stride_labels[1])
            losses['wh'] += self.scale_loss(stride_out[1], stride_labels[1], mask=pos_mask)
            losses['of'] += self.offset_loss(stride_out[2], stride_labels[2], mask=pos_mask)

        probes /= num_maps
        for key in losses.keys():
            losses[key] /= num_maps

        return losses, probes


if __name__ == "__main__":
    from utils.utils import load_yaml
    from factory import get_fpn_net
    
    cfg = load_yaml('config.yaml')
    net = get_fpn_net(cfg['net'])

    x1 = torch.randn(4, 3, 320, 320)
    x2 = torch.randn(4, 3, 320, 320)

    out1 = net(x1)
    out2 = net(x2)

    crit = AnchorFreeLoss(cfg['train'])

    losses, probs = crit(out1, out2)
    print(losses, probs)