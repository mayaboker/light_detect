from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

def test(net, dataset, heads=None, batch_size=8):
    net.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False
    )

    threshold = 0.05
    iou_thresh = 0.4
    thresh_num = 1000
    count_obj = 0

    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    for i, data in tqdm(enumerate(loader), desc="Test: "):
        pass
