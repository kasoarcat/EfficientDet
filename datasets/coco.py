from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import cv2
from PIL import Image
import h5py

class H5CoCoDataset(torch.utils.data.Dataset):
    def __init__(self, path, set_name):
        self.path = path
        self.set_name = set_name
        self.index = 0
        with h5py.File(self.path, 'r') as f:
            self.len = len(f["img"])
            self.image_ids = [int(i) for i in f['image_ids']]
            self.coco_labels = dict(zip(f['coco_labels_keys'], f['coco_labels_values']))

    def __getitem__(self, index):
        img = None
        annot = None
        with h5py.File(self.path, 'r') as f:
            img = torch.tensor(f["img"][str(index)])
            annot = torch.tensor(f["annot"][str(index)])
            scale = np.array(f["scale"][str(index)])
        return {'img':img, 'annot':annot, 'scale':scale}
    
    def __len__(self):
        return self.len
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < self.len:
            item = self[self.index]
            self.index += 1
            return item
        else:
            self.index = 0
            raise StopIteration
            
    def image_aspect_ratio(self, image_index):
        img = self[image_index]['img']
        return float(img.shape[0]) / float(img.shape[1])

    def label_to_coco_label(self, label):
        return int(self.coco_labels[label])
        
class CocoDataset(Dataset):
    """Coco dataset."""
    def __init__(self, root_dir, set_name, transform=None, limit_len=0):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.coco = COCO(os.path.join(self.root_dir, self.set_name, self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.load_classes()
        if limit_len == 0:
            self.len = len(self.image_ids)
        else:
            self.len = limit_len
        self.index = 0

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.len:
            item = self[self.index]
            self.index += 1
            return item
        else:
            self.index = 0
            raise StopIteration

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample[annot] = self.transform(**sample[annot])
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        # path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        path = os.path.join(self.root_dir, self.set_name, 'images', image_info['file_name'])
        # print('path:[%s]' % (path))
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

if __name__ == '__main__':
    from augmentation import get_augumentation
    dataset = CocoDataset(root_dir='/root/data/coco', set_name='trainval35k',
                          transform=get_augumentation(phase='train'))
    sample = dataset[0]
    print('sample: ', sample)
