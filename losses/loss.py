import torch.nn as nn
import torch



class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss(reduction='sum')

    def forward(self, pred, gt, mask=None):
        if mask is not None:
            pred = pred[mask]
            gt = gt[mask]
        num = pred.numel()
        if num == 0:
            return 0
        loss = self.loss(pred, gt)
        loss = loss / num
        return loss
        

class ASLoss(nn.Module):
    def __init__(self, gamma_p=2, gamma_n=4, margin=0.05, beta=4, use_negative_weights=True):
        super(ASLoss, self).__init__()
        self.gamma_p = gamma_p
        self.gamma_n = gamma_n
        self.marg = margin
        self.beta = beta
        self.use_negative_weights = use_negative_weights
        self.eps = 1e-5

    def forward(self, pred, gt):
        pred = torch.clamp(pred, self.eps, 1 - self.eps)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        if self.use_negative_weights:
            neg_weights = torch.pow(1-gt, self.beta)
        else:
            neg_weights = 1

        neg_pred = (pred - self.marg).clamp(min=self.eps)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.gamma_p) * pos_inds
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, self.gamma_n) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        p_prob = pred[torch.where(gt == 1)].mean().item()
        n_prob = (1 - pred[torch.where(gt < 0.1)]).mean().item()

        if num_pos == 0:
            loss = - neg_loss
        else:
            loss = - (pos_loss + neg_loss)  / num_pos
        
        return loss, torch.tensor([p_prob, n_prob])

def FocalLoss(gamma=2, beta=4, use_negative_weights=True):
    return ASLoss(gamma, gamma, 0, beta, use_negative_weights)


if __name__ == "__main__":
    myloss = ASLoss()
    predict = torch.randn(32, 1, 200, 200)
    predict = 1 / (1 + torch.exp(-predict))

    groundTruth = torch.randn(32, 1, 200, 200)
    groundTruth = 1 / (1 + torch.exp(-groundTruth))
    print(myloss.forward(predict, groundTruth)[0])





