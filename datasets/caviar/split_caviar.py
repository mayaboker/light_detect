import os
import json
from pathlib import Path
from shutil import move
import numpy as np
import glob

if __name__ == "__main__":
    root = '/home/core4/data/CAVIAR'
    folders = glob.glob(root + '/**/**/')    
    i_rand = np.random.permutation(len(folders))    
    n_train = int(0.7*len(folders))
    n_val = int(0.2*len(folders))
    i_train = i_rand[:n_train]
    i_val = i_rand[n_train:n_train+n_val]    
    i_test = i_rand[n_train+n_val:]
    
    train_folders = [folders[i] for i in i_train]
    val_folders = [folders[i] for i in i_val]
    test_folders = [folders[i] for i in i_test]

    data_parsed = {}
    for i, d in enumerate(train_folders):
        fname = d[:-1] + '_gt.json'
        with open(fname, "r") as f: 
            data = json.load(f)
        data_parsed.update(data)        
    
    with open('/home/core4/data/CAVIAR/train_gt.json', "w") as f: 
        data = json.dump(data_parsed, f)

    
    data_parsed = {}
    for i, d in enumerate(val_folders):
        fname = d[:-1] + '_gt.json'
        with open(fname, "r") as f: 
            data = json.load(f)
        data_parsed.update(data)        
    
    with open('/home/core4/data/CAVIAR/val_gt.json', "w") as f: 
        data = json.dump(data_parsed, f)

    data_parsed = {}
    for i, d in enumerate(test_folders):
        fname = d[:-1] + '_gt.json'
        with open(fname, "r") as f: 
            data = json.load(f)
        data_parsed.update(data)    
    
    with open('/home/core4/data/CAVIAR/test_gt.json', "w") as f: 
        data = json.dump(data_parsed, f)    