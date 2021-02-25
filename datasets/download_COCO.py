from pycocotools.coco import COCO
import shutil
import json


def download(annotations_file, images_file, subimages_file):
    categories = ['truck', 'car', 'person']
    coco = COCO(annotations_file)
    images = []
    with open(annotations_file, 'r') as f:
        data = json.load(f)
        for key in data:
            print(key)

    for cat in categories:
        catIds = coco.getCatIds(catNms=[cat])
        imgIds = coco.getImgIds(catIds=catIds)
        images += coco.loadImgs(imgIds)

    for img in images:
        image_file_name = img['file_name']
        shutil.copy(images_file + "/" + image_file_name, subimages_file)


if __name__ == '__main__':
    annotations_file = r'/home/kobi/PycharmProjects/vehicle_recognition_coco/annotations/instances_train2017.json'
    images_file = r"/home/kobi/Documents/train2017"
    subimages_file = r"/home/kobi/PycharmProjects/vehicle_recognition_coco/subdataset images"
    download(annotations_file, images_file, subimages_file)
