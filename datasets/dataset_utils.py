import torch
import math
import numpy as np
from utils.image_utils import gaussian_radius, draw_umich_gaussian


def make_heatmaps(im, bboxes, stride):
    height, width = im.size()[1:]
    out_width = int(width / stride)
    out_height = int(height / stride)

    res = np.zeros([5, out_height, out_width], dtype=np.float16)
    for bbox in bboxes:
        #0 heatmap
        left, top, right, bottom = map(lambda x: x / stride, bbox[:4])
        if right <= left or bottom <= top:
            continue
        x = int((left + right) /2)
        y = int((top + bottom) /2)

        x, y = min(max(x, 0), out_width-1), min(max(y, 0), out_height-1)
        h, w = (bottom - top), (right - left)

        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(int(radius), 0)
        draw_umich_gaussian(res[0], (x, y), radius)

        # 1, 2 center offset
        center_x = (left + right) / 2
        center_y = (top + bottom) / 2
        res[1][y, x] = center_x - x
        res[2][y, x] = center_y - y

        # 3, 4 width and height
        eps = 1e-6
        res[3][y, x] = np.log(w + eps)
        res[4][y, x] = np.log(h + eps)

    return res


def test_collate_fn(batch):
    in_tensor = []
    boxes = []
    for b in batch:
        in_tensor.append(b[0].unsqueeze(0))
        boxes.append(b[1])
    in_tensor = torch.cat(in_tensor, 0)
    return in_tensor, boxes