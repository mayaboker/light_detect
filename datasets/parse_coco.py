from torch.utils.data import Dataset
from PIL import Image
import json
import glob
import re


class COCODataset(Dataset):
    def __init__(self, json_file, images_path, transforms=None):
        file_data = json_file
        self.labels = []
        self.f_imgs = glob.glob(images_path + '/*.jpg')
        with open(file_data, 'r') as f:
            data_instances = json.load(f)
            for img in self.f_imgs:
                img_name = int(re.search(r"\d+[.jpg]", img).group()[:-1])
                same_id_samples = []
                for sample in data_instances["annotations"]:
                    if sample["image_id"] == img_name:
                        same_id_samples.append(sample["category_id"])
                self.labels.append(same_id_samples)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        labels = self.labels[index]
        img = Image.open(self.f_imgs[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, labels


if __name__ == '__main__':
    annotations_file = r'/home/kobi/PycharmProjects/vehicle_recognition_coco/annotations/instances_train2017.json'
    subimages_file = r"/home/kobi/PycharmProjects/vehicle_recognition_coco/subdataset images"
    coco = COCODataset(annotations_file, subimages_file)
    print('hi')
