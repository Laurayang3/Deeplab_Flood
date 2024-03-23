#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30

from __future__ import absolute_import, print_function

import os.path as osp
from glob import glob

import cv2
import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils import data
import glob
import sys
import os
sys.path.append(os.getcwd())

from libs.datasets.base import _BaseDataset


class FloodNet(_BaseDataset):
    """FloodNet dataset"""

    def __init__(self, warp_image=True, **kwargs):
        self.warp_image = warp_image
        super(FloodNet, self).__init__(**kwargs)

    def _set_files(self):
        # Create data list via {train, test, all}.txt
        if self.split in ["train", "test", "val"]:
            file_path = osp.join(self.root, self.split, f"{self.split}-org-img")
            jpg_images = []
            for root, dirs, files in os.walk(file_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg')):
                        jpg_images.append(file.split(".")[0])
            self.files = jpg_images
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, index):
        # Set paths
        image_id = self.files[index]
        image_path = osp.join(self.root, self.split, f"{self.split}-org-img", image_id + ".jpg")
        label_path = osp.join(self.root, self.split, f"{self.split}-label-img", image_id + "_lab.png")
        # Load an image and label
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = label.astype(np.uint8)
        # label -= 1  # unlabeled (0 -> -1)
        # label[label == -1] = 255
        # Warping: this is just for reproducing the official scores on GitHub
        if self.warp_image:
                
            image = cv2.resize(image, (513, 513), interpolation=cv2.INTER_LINEAR)

            label = cv2.resize(label, (513, 513), interpolation=cv2.INTER_LINEAR)

            label = np.asarray(label, dtype=np.int32)
        return image_id, image, label




def get_parent_class(value, dictionary):
    # Get parent class with COCO-Stuff hierarchy
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    import yaml
    from torchvision.utils import make_grid
    from tqdm import tqdm

    kwargs = {"nrow": 10, "padding": 50}
    batch_size = 100

    dataset = FloodNet(
        root="/shared/user82/workspace/myang3/data/FloodNet",
        split="train",
        ignore_label=255,
        mean_bgr=(104.008, 116.669, 122.675),
        augment=True,
        crop_size=321,
        scales=(0.5, 0.75, 1.0, 1.25, 1.5),
        flip=True,
    )
    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, (image_ids, images, labels) in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):
        if i == 0:
            mean = torch.tensor((104.008, 116.669, 122.675))[None, :, None, None]
            images += mean.expand_as(images)
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 255
            image = np.dstack((image, mask)).astype(np.uint8)

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=255, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 182.0) * 255
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 255)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            # cv2.imwrite("./docs/datasets/cocostuff.png", tiled_images)
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break

    class_hierarchy = "./data/datasets/cocostuff/cocostuff_hierarchy.yaml"
    data = yaml.load(open(class_hierarchy))
    key = "person"

    for _ in range(3):
        key = get_parent_class(key, data)
        key = list(key)[0]
        print(key)
