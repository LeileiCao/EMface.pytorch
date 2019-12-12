#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random

class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file):
        super(WIDERDetection, self).__init__()
        self.fnames = []
        self.boxes = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
                 
        return torch.from_numpy(np.array(boxes).astype(np.float32))
        

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
