import torch
from pathlib import Path
import pandas as pd
from xml.dom import minidom


def preprocess_voc2012(path, categories):
    annotations_path = path / 'Annotations'
    all_annotations = annotations_path.glob('**/*.xml')
    filenames = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    objects_category = []
    for annotation_path in all_annotations:
        annotation_path_str = str(annotation_path)
        curr_annotation = minidom.parse(annotation_path_str)
        filename = curr_annotation.getElementsByTagName('filename').item(0).firstChild.data
        objects = curr_annotation.getElementsByTagName('object')
        for curr_object in objects:
            category = curr_object.getElementsByTagName('name').item(0).firstChild.data
            if category in categories:
                bounding_box = curr_object.getElementsByTagName('bndbox')
                xmin = bounding_box[0].getElementsByTagName('xmin').item(0).firstChild.data
                ymin = bounding_box[0].getElementsByTagName('ymin').item(0).firstChild.data
                xmax = bounding_box[0].getElementsByTagName('xmax').item(0).firstChild.data
                ymax = bounding_box[0].getElementsByTagName('ymax').item(0).firstChild.data
                filenames.append(filename)
                xmins.append(xmin)
                xmaxs.append(xmax)
                ymins.append(ymin)
                ymaxs.append(ymax)
                objects_category.append(category)
    data = {
        'filename': filenames,
        'category': objects_category,
        'xmin': xmins,
        'ymin': ymins,
        'xmax': xmaxs,
        'ymax': ymaxs
    }
    df = pd.DataFrame(data=data, columns=['filename', 'category', 'xmin', 'ymin', 'xmax', 'ymax'])

    print(df.head())

    df.to_csv(path / 'voc_data.csv')
        

if __name__ == '__main__':
    voc_root = '/home/core4/data/pascal_voc_seg/VOCdevkit/VOC2012'
    path_to_annotations = Path(voc_root)
    categories = ['person', 'bus', 'car']
    preprocess_voc2012(path_to_annotations, categories)


# class VOC2012Dataset(torch.utils.data.Dataset):
#     pass