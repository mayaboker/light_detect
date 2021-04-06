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
    f_model = '/home/core4/Documents/logs/train_virat/wh05/checkpoints/Epoch_47.pth'  
    fnames = glob.glob('/home/core4/data/S1-20210405T161314Z-001/S1/Videos/*.avi')               
    fnames = [r'/home/core4/data/S1-20210405T161314Z-001/S1/Videos/PRG22.avi']
    height = 320
    width = 320
    '''model'''
    cfg = load_yaml('config.yaml')
    net = get_fpn_net(cfg['net'])    
    sd = torch.load(f_model)['net_state_dict']
    net.load_state_dict(sd)    
    net.cuda()

    for i_file, fname in enumerate(fnames):
        cap = cv2.VideoCapture(fname.__str__())
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        i = 0
        f_res = fname[:-4] + '_res.avi'
        print(fname)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid_out = cv2.VideoWriter(f_res,fourcc, fps, (width, height))
        while(True):
            if i % 100 == 0:
                print("processing {}/{}".format(i, n_frames))
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break            
            frame, _ = padded_resize(frame, size=(320, 320))
            x = to_tensor(frame).unsqueeze(0).float()
            x = normalize(x, mean=[0]*3, std=[1]*3)
            x = x.cuda()                
            with torch.no_grad():
                out = net(x)
            boxes, scores = decode(out, (4,), 0.4, K=50, use_nms=True)
            #print(len(boxes[0]))               
            for j, (l, s) in enumerate(zip(boxes[0], scores[0])):                    
                frame = cv2.rectangle(frame, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 1)                
            vid_out.write(frame)
            i = i + 1            
        cap.release()
        vid_out.release()
        cv2.destroyAllWindows()                