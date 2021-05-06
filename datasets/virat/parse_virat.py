import cv2
import glob
from pathlib import Path
import xmltodict, json
import matplotlib.pyplot as plt
import os 
import numpy as np
import math

MAX_SIZE = 640

def display_data(frame, boxes):
    for bbox in boxes:
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 1)
    plt.imshow(frame)
    plt.show()

def parse_ann(anns):
    with open(anns, 'r') as f:
        lines = f.readlines()
    labels = {}
    for line in lines:
        data = line.strip().split(' ')
        type_id = data[-1]

        if type_id != '1': # if not a person
            continue
        num_frame = int(data[2])
        xmin = float(data[3])
        ymin = float(data[4])
        w = float(data[5])
        h = float(data[6])
        xmax = xmin + w
        ymax = ymin + h

        if num_frame not in labels:
            labels[num_frame] = []
        labels[num_frame].append([xmin, ymin, xmax, ymax, float(type_id)])

    return labels


def resize_frame(frame, max_size, boxes):
    h, w = frame.shape[:2]
    big_size = max(h, w)
    scale = big_size / max_size
    
    dst_h, dst_w = int(h / scale), int(w / scale)
    frame = cv2.resize(frame, (dst_w, dst_h))

    new_boxes = []
    for box in boxes:
        box = [b / scale for b in box] 
        new_boxes.append(box)
    return frame, new_boxes

def parse_data(root, out_dir, fps_save=2):
    fnames_mpg = sorted(glob.glob(root + '/videos/*.mp4'))
    fnames_ann = sorted(glob.glob(root + '/annotations/*.txt'))

    for fname in fnames_mpg:            
        v_name = fname.split('/')[-1].split('.')[0]                                   
        print(v_name)

        anns = [f for f in fnames_ann if v_name in f and 'objects' in f][0]
        labels = parse_ann(anns)   
        v_labels = {}
        
        cap = cv2.VideoCapture(fname)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        frames_path = os.path.join(root, out_dir, v_name)
        Path(frames_path).mkdir(exist_ok=True, parents=True)
        i = 0
        while(True):
            if i % 100 == 0:
                print("processing {}/{}".format(i, n_frames))
            # Capture frame-by-frame
            ret, frame = cap.read()
            i = i + 1
            if not ret:
                break
            if i % (fps // fps_save) != 0 or i not in labels.keys():
                continue
            
            boxes = labels[i]
            frame, boxes = resize_frame(frame, MAX_SIZE, boxes)

            frame_path = os.path.join(frames_path, str(i).zfill(6) + '.png')
            frame_name = os.path.join(v_name, str(i).zfill(6) + '.png')
            cv2.imwrite(frame_path, frame)
            
            v_labels[frame_name] = boxes   

        labels_path = os.path.join(frames_path, 'labels_gt.json')
        with open(labels_path, "w") as write_file:
                json.dump(v_labels, write_file)
        cap.release()
    

def load_json(j_path):
    with open(j_path, 'r') as j:
        data = json.load(j)
    return data

def save_json(j_path, data):
    with open(j_path, 'w') as j:
        json.dump(data, j)


def extract_scenes(video_names):
    scenes = {}
    for i, name in enumerate(video_names):
        scene = name.split('/')[-1].split('_')[2]
        if scene not in scenes.keys():
            scenes[scene] = []
        scenes[scene].append(i)
    return scenes

def split_data(root):
    subsets = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    # scenes = extract_scenes(subsets)

    train_set = {}
    val_set = {}
    test_set = {}

    inds = np.arange(0, len(subsets), 1)
    rands = np.random.permutation(inds)

    split_inds = [int(len(rands)*0.7), int(len(rands)*0.9)]
    
    train_set = {}
    val_set = {}
    test_set = {}
    for i, ind in enumerate(rands):
        if i < split_inds[0]:
            print(f'Train: #{i} {subsets[ind]}')
            train_set.update(
                load_json(os.path.join(subsets[ind], 'labels_gt.json'))
            )
        elif i < split_inds[1]:
            print(f'Val: #{i} {subsets[ind]}')
            val_set.update(
                load_json(os.path.join(subsets[ind], 'labels_gt.json'))
            )
        else:
            print(f'Test: #{i} {subsets[ind]}')
            test_set.update(
                load_json(os.path.join(subsets[ind], 'labels_gt.json'))
            )
    save_json(os.path.join(root, 'train_gt.json'), train_set)
    save_json(os.path.join(root, 'val_gt.json'), val_set)
    save_json(os.path.join(root, 'test_gt.json'), test_set)
    print(f'Train samples: {len(train_set)}')
    print(f'Validation samples: {len(val_set)}')
    print(f'Test samples: {len(test_set)}')

if __name__ == "__main__":
    root = r'/media/robert/DATA/virat'
    out_dir = 'full'
    parse_data(root, out_dir, fps_save=2)
    split_data(os.path.join(root, out_dir))

