from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
from api import decode
from evaluate.evaluation import image_eval, img_pr_info, dataset_pr_info, voc_ap
from datasets.dataset_utils import test_collate_fn
import cv2 
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def test(net, dataset, batch_size=32):
    net.eval()

    strides = dataset.strides

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=0,
        shuffle=False,
        collate_fn=test_collate_fn
    )

    threshold = 0.05
    iou_thresh = 0.4
    thresh_num = 1000
    count_obj = 0
    
    if False:
        for i, data in tqdm(enumerate(loader), desc="Test: ", ascii=True, total=len(loader)):
            img, labels = data
            img = img.cuda()
            with torch.no_grad():
                out = net(img)
            boxes, scores = decode(out, strides, 0.35, K=100)            
            img_show = np.array(to_pil_image(img.squeeze(0).cpu()))
            for ll in labels:
                for l in ll:
                    img_show = cv2.rectangle(img_show, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (255, 0, 0), 1)
            for i, l in enumerate(boxes[0]):
                img_show = cv2.rectangle(img_show, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 1)
        
            plt.imshow(img_show)
            plt.show()
        
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    for i, data in tqdm(enumerate(loader), desc=f"Test-{dataset.name()}: ", ascii=True, total=len(loader)):
        img, labels = data
        img = img.cuda()
        with torch.no_grad():
            out = net(img)
        boxes, scores = decode(out, strides, threshold, K=100)                

        for i in range(len(labels)):
            gt_boxes = labels[i].astype(np.double)
            result = []
            for b, s in zip(boxes[i], scores[i]):
                x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
                box = [x1, y1, x2 - x1 +1, y2 - y1 +1, s]
                box = np.array(box).astype(np.double)
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
    from transformations import get_test_transforms
    from factory import get_fpn_net, get_pedestrian_dataset

    cfg = load_yaml('config.yaml')
    dataset = get_pedestrian_dataset(
                'virat',
                cfg['paths'],
                augment=get_test_transforms(cfg['train']['transforms']),
                mode='test'
    )

    net = get_fpn_net(cfg['net'])
    net.cuda()
    sd = torch.load('../logs/train_caviar_virat/a/checkpoints/Epoch_104.pth')['net_state_dict']
    net.load_state_dict(sd)
    ap = test(net, dataset)
    print(ap)