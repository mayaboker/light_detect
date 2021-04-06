import torch
import math
import numpy as np
from torchvision.transforms.functional import scale
from utils.image_utils import gaussian_radius, draw_umich_gaussian
import cv2

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

def padded_resize(im, size, bboxes=None):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes.copy())
    canvas = np.zeros((size[0], size[1], 3), dtype=np.uint8)#Image.new('RGB', size=size, color="#777")
    target_width, target_height = size
    height, width = im.shape[:2]
    offset_x = 0
    offset_y = 0
    if height > width:
        height_ = target_height
        scale = height_ / height
        width_ = int(width * scale)
        offset_x = (target_width-width_)//2
    else:
        width_ = target_width
        scale = width_ / width
        height_ = int(height * scale)
        offset_y = (target_height-height_)//2
    im = cv2.resize(im, dsize=(width_, height_))
    canvas[offset_y:im.shape[0]+offset_y, offset_x:im.shape[1]+offset_x] = im
    if bboxes is not None:# and len(bboxes) == 0:
        bboxes = bboxes.copy()
        bboxes *= scale
        bboxes[:, 0::2] += offset_x
        bboxes[:, 1::2] += offset_y
        bboxes = bboxes.tolist()
    return canvas, bboxes

if __name__ == "__main__":
    f_name = '/home/core4/data/virat/frames/VIRAT_S_000200_00_000100_000171/000645.png'
    import cv2
    im = cv2.imread(f_name)
    imm, _ = padded_resize(im, (320, 320))
    cv2.imshow('1', im)
    cv2.imshow('2', imm)
    cv2.waitKey(0)
