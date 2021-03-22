import os
import json
from pathlib import Path
from shutil import move

if __name__ == "__main__":
    root = '/home/core4/data/CAVIAR'
    train_folders = ['Browse1', 'Browse2', 'Browse3', 'Browse4', 'Browse_WhileWaiting1', 'Browse_WhileWaiting2', 
                     'Rest_FallOnFloor', 'Walk1', 'Walk2']
    val_folders = ['Rest_InChair', 'Rest_SlumpOnFloor', 'Walk3']
    test_folders = ['Rest_WiggleOnFloor', 'Rest_FallOnFloor']
    
    for i, d in enumerate(train_folders):        
        if os.path.isdir(os.path.join(root, d)):
            move(os.path.join(root, d), os.path.join(root, 'train', d))

    for i, d in enumerate(val_folders):
        if os.path.isdir(os.path.join(root, d)):
            move(os.path.join(root, d), os.path.join(root, 'val', d))

    for i, d in enumerate(test_folders):        
        if os.path.isdir(os.path.join(root, d)):
            move(os.path.join(root, d), os.path.join(root, 'test', d))
        
    data_parsed = {}
    for i, d in enumerate(train_folders):
        fname = os.path.join(root, d) + '_gt.json'
        with open(fname, "r") as f: 
            data = json.load(f)
        data_parsed.update(data)    
    
    with open('train_gt.json', "w") as f: 
        data = json.dump(data_parsed, f)

    data_parsed = {}
    for i, d in enumerate(val_folders):
        fname = os.path.join(root, d) + '_gt.json'
        with open(fname, "r") as f: 
            data = json.load(f)
        data_parsed.update(data)    
    
    with open('val_gt.json', "w") as f: 
        data = json.dump(data_parsed, f)

    data_parsed = {}
    for i, d in enumerate(test_folders):
        fname = os.path.join(root, d) + '_gt.json'
        with open(fname, "r") as f: 
            data = json.load(f)
        data_parsed.update(data)    
    
    with open('test_gt.json', "w") as f: 
        data = json.dump(data_parsed, f)
    
    '''
    forig = r"/home/core4/data/CAVIAR/Browse1_gt.json"
    fres = r"/home/core4/data/CAVIAR/Browse1-debug_gt.json"
    
    with open(forig, "r") as f: 
        data = json.load(f)
    data_parsed = {}
    
    for k in range(16):
        data_parsed[str(k)] = data[str(k)]
    
    with open(fres, "w") as f: 
        data = json.dump(data_parsed, f)
    '''