import cv2
import glob
from pathlib import Path
import xmltodict, json
import matplotlib.pyplot as plt
import os 
import numpy as np
import xml.etree.ElementTree as ET


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
        labels[num_frame].append([xmin, ymin, xmax, ymax, type_id])

    return labels

def parse_data(root, fps_save=2):
    fnames_mpg = sorted(glob.glob(root + '/Videos/*.avi'))
    #fnames_ann = sorted(glob.glob(root + '/TrackletsUsed/*.xml'))

    for fname in fnames_mpg:           
        v_name = fname.split('/')[-1].split('.')[0]        
        anns = sorted(glob.glob(root + '/TrackletsUsed/Cam ' + v_name[3:] + '/*.xml'))
        v_labels = {}        
        for i, ann in enumerate(anns):
            print(ann)
            tree = ET.parse(ann)
            xml_data = tree.getroot()        
            xmlstr = ET.tostring(xml_data, encoding='utf-8', method='xml')
            data_dict = dict(xmltodict.parse(xmlstr))
            objs = data_dict['TrackletInfo']['Frame']
            frames = []
            for j, obj in enumerate(objs):
                frame_idx = int(obj['@Number'])
                h = int(obj['Height'])
                w = int(obj['Width'])
                x_min = float(obj['Left_X'])
                y_min = float(obj['Left_Y'])
                x_max = x_min + w - 1
                y_max = y_min + h - 1
                bbox = [x_min, y_min, x_max, y_max, 1]                
                f_name = str(frame_idx).zfill(6)
                v_labels[f_name] = [bbox]
                frames.append(frame_idx)               
        cap = cv2.VideoCapture(fname)
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frames_path = os.path.join(root, 'frames', v_name)
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
            if i % (fps // fps_save) != 0 or i not in frames:
                continue
            frame_path = os.path.join(frames_path, str(i).zfill(6) + '.png')
            #frame_name = os.path.join(v_name, str(i).zfill(6) + '.png')
            l = v_labels[str(i).zfill(6)][0]
            frame = cv2.rectangle(frame, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 255, 0), 1)
            cv2.imwrite(frame_path, frame)                     

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

def split_data(root):
    subsets = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    rands = np.random.permutation(subsets)

    split_inds = [int(len(rands)*0.7), int(len(rands)*0.9)]
    
    train_set = {}
    val_set = {}
    test_set = {}
    for i, subset in enumerate(subsets):
        if i < split_inds[0]:
            train_set.update(
                load_json(os.path.join(subset, 'labels_gt.json'))
            )
        elif i < split_inds[1]:
            val_set.update(
                load_json(os.path.join(subset, 'labels_gt.json'))
            )
        else:
            test_set.update(
                load_json(os.path.join(subset, 'labels_gt.json'))
            )
    save_json(os.path.join(root, 'train_gt.json'), train_set)
    save_json(os.path.join(root, 'val_gt.json'), val_set)
    save_json(os.path.join(root, 'test_gt.json'), test_set)
    print(f'Train samples: {len(train_set)}')
    print(f'Validation samples: {len(val_set)}')
    print(f'Test samples: {len(test_set)}')

if __name__ == "__main__":
    root = r'/home/core4/data/S1-20210405T161314Z-001/S1'
    parse_data(root, fps_save=1)
    #split_data(os.path.join(root, 'frames'))

