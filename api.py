import numpy as np
from torchvision.ops import nms
import torch
import torch.nn.functional as F
from torchvision.ops import nms
def decode(net_outputs, strides, threshold, K, nms_type='union'):
    # net_outputs: List[ List[Tensor] ... List[Tensor]]

    all_bboxes = torch.Tensor([]).cuda()
    all_scores = torch.Tensor([]).cuda()
    for output, stride in zip(net_outputs, strides):
        hm = output[0]
        of = output[1]
        wh = output[2]

        hm = pool_nms(hm)

        batch, classes, height, width = hm.size()
        scores, inds, clses, ys, xs = _topk(hm, K)

        #offset
        off = _transpose_and_gather_feat(of, inds)
        off = off.view(batch, K, 2)
        cx = xs.view(batch, K, 1) + off[:, :, 0:1]
        cy = ys.view(batch, K, 1) + off[:, :, 1:2]

        #scales
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        wh = torch.exp(wh)

        #print(clses.shape, scores.shape)
        # clses = clses.view(batch, K, 1).float()
        # scores = scores.view(batch, K, 1)

        bboxes = torch.cat([
            cx - wh[..., 0:1] /2,
            cy - wh[..., 1:2] /2,
            cx + wh[..., 0:1] /2,
            cy + wh[..., 1:2] /2,
        ], dim=2)

        bboxes = bboxes * stride
        # append results
        all_bboxes = torch.cat((all_bboxes, bboxes), dim=1)
        all_scores = torch.cat((all_scores, scores), dim=1)
    
    bboxes = []
    scores = []
    for i in range(all_bboxes.size(0)):
        keep = nms(all_bboxes[i], all_scores[i], iou_threshold=0.4)
        bboxes.append(all_bboxes[i, keep].cpu().numpy())
        scores.append(all_scores[i, keep].cpu().numpy())
    
    bboxes, scores = _filter_by_threshold(
        bboxes,
        scores,
        threshold
    )    
    
    return bboxes, scores


def _filter_by_threshold(bboxes, scores, threshold):
    l_boxes, l_scores = [], []
    for bs, ss in zip(bboxes, scores):
        inds = np.where(ss >= threshold)[0]
        l_scores.append(ss[inds])
        l_boxes.append(bs[inds])
    return l_boxes, l_scores



def _topk(scores, K):
    batch, classes, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, classes, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind
    ).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def pool_nms(x, kernel=3):
    pad = kernel // 2
    hmax = F.max_pool2d(x, kernel, stride=1, padding=pad)
    keep = torch.floor(x - hmax + 1)
    return x * keep

