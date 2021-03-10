from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from api import decode
from evaluate.evaluation import image_eval, img_pr_info, dataset_pr_info, voc_ap
from datasets.dataset_utils import test_collate_fn

def test(net, dataset, heads=None, batch_size=8):
    net.eval()

    strides = dataset.strides

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=True,
        collate_fn=test_collate_fn
    )

    threshold = 0.05
    iou_thresh = 0.4
    thresh_num = 1000
    count_obj = 0

    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    for i, data in tqdm(enumerate(loader), desc="Test: ", ascii=True, total=len(loader)):
        img, labels = data
        img = img.cuda()
        with torch.no_grad():
            out = net(img)
        boxes, scores = decode(out, strides, threshold, K=100)

        for i in range(len(labels)):
            gt_boxes = labels[i]
            result = []
            for b, s in zip(boxes[i], scores[i]):
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                box = [x1, y1, x2 - x1 +1, y2 - y1 +1, s]
                box = np.array(box).astype('float')
                result.append(box)
            result = np.array(result)
            count_obj += len(gt_boxes)
            if len(gt_boxes) == 0 or len(result) == 0:
                continue
            ignore = np.ones(gt_boxes.shape[0])
            pred_recall, proposal_list = image_eval(result, gt_boxes, ignore, iou_thresh, box_format='xyxy')
            _img_pr_info = img_pr_info(thresh_num, result, proposal_list, pred_recall)
            pr_curve += _img_pr_info

    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_obj)
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    ap = voc_ap(recall, propose)
    return ap




if __name__ == "__main__":
    from utils.utils import load_yaml
    from datasets.CAVIARDataset import CAVIARDataset
    from transformations import get_test_transforms
    from factory import get_fpn_net

    cfg = load_yaml('config.yaml')
    dataset = CAVIARDataset(cfg['paths']['data_dir'], augment=get_test_transforms(cfg['train']['transforms']), mode='test', strides=cfg['net']['strides'])

    net = get_fpn_net(cfg['net'])
    net.cuda()

    sd = torch.load('../logs/test2/checkpoints/Epoch_199.pth')['net_state_dict']
    net.load_state_dict(sd)

    ap = test(net, dataset)
    print(ap)


        