from datasets.dataset_utils import padded_resize
import torch
import cv2
from torch.autograd.grad_mode import F
from utils.utils import load_yaml
import glob
import os
import json
from factory import get_fpn_net, get_pedestrian_dataset
from transformations import get_test_transforms
from api import decode
from torchvision.transforms.functional import to_pil_image, to_tensor, normalize
from PIL import Image
import numpy as np

if __name__ == "__main__":    
    
    '''model'''
    f_model = '/home/core4/Documents/logs/train_virat/wh05_160/checkpoints/Epoch_199.pth'    
    root = '/home/core4/data/virat/frames/VIRAT_S_010200_10_000923_000959' # test
    root = '/home/core4/data/virat/frames/VIRAT_S_010201_07_000601_000697'
    #root = '/home/core4/data/CAVIAR'
    ext = 'png'    
    
    RUN_MODEL = True
    RUN_GT = False
    height = 320#288#720
    width = 320#384#1280
    # if RUN_GT:
    #     with open(os.path.join(root, "labels_gt.json")) as f:
    #         test_gt = json.load(f)    

    if RUN_MODEL:
        '''model'''
        cfg = load_yaml('config.yaml')
        net = get_fpn_net(cfg['net'])    
        sd = torch.load(f_model)['net_state_dict']
        net.load_state_dict(sd)    
        net.cuda()
    
    #clips = sorted(set([t.split('/')[0] + '/' + t.split('/')[1] for t in test_gt.keys()]))
    clips = [root]
    for i, c in enumerate(clips):        
        f_res = os.path.join(root, c) + '.avi'
        print(f_res)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_out = cv2.VideoWriter(f_res,fourcc, 3, (width, height))

        f_frames = sorted(glob.glob(os.path.join(root, c) + '/*.'+ext))
        if RUN_GT:
            # gt_name = os.path.join(os.path.join(root, c) + '_gt.json')
            gt_name = os.path.join(root, 'labels_gt.json')
            with open(gt_name) as f:
                clip_gt = json.load(f)
        for j, f_frame in enumerate(f_frames):
            frame = np.array(Image.open(f_frame))            
            txt = os.path.basename(f_frame).split('.')[0]
            frame = cv2.putText(frame, txt, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if RUN_GT:
                frame_name = os.path.join(os.path.basename(c), os.path.basename(f_frame))
                if frame_name in clip_gt.keys():
                    bbox_gt = clip_gt[frame_name]
                    for i, l in enumerate(bbox_gt):
                        frame = cv2.rectangle(frame, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 0, 255), 2)
            if RUN_MODEL:                
                frame, _ = padded_resize(frame, size=(320, 320))#cv2.resize(x, dsize=(320, 320))
                x = to_tensor(frame).unsqueeze(0).float()
                x = normalize(x, mean=[0]*3, std=[1]*3)
                x = x.cuda()                
                with torch.no_grad():
                    out = net(x)
                boxes, scores = decode(out, (4,), 0.4, K=50, use_nms=True)
                print(len(boxes[0]))               
                for i, (l, s) in enumerate(zip(boxes[0], scores[0])):                    
                    #l[0] = l[0] * sx
                    #l[2] = l[2] * sx
                    #l[1] = l[1] * sy
                    #l[3] = l[3] * sy
                    frame = cv2.rectangle(frame, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 1)                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imshow('', frame)
            vid_out.write(frame)
        vid_out.release()
        
