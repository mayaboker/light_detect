import json
import numpy as np
import glob
import matplotlib.pyplot as plt


if __name__ == "__main__":
    root = '/home/core4/data/CAVIAR'
    fnames = glob.glob(root + '/*.json')
    w = []
    h = []
    for i, f in enumerate(fnames):
        with open(f, 'r') as f:
            data = json.load(f)
        for i, k in enumerate(data.keys()):            
            l = data[k]['bbox']
            labels = [ll[:5] for ll in l]
            for l in labels:
                w.append(l[2]-l[0])
                h.append(l[3]-l[1])
    w = np.array(w)
    h = np.array(h)
    print(w.shape[0], h.shape[0])
    plt.subplot(2,1,1)
    plt.hist(w, bins=50)
    plt.subplot(2,1,2)
    plt.hist(h, bins=50)
    plt.show()    
