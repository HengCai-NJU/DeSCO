# -*- coding: utf-8 -*-
import h5py, os
import torch, cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose


class make_data(Dataset):
    def __init__(self, img, lab, mask):
        self.img = img
        self.lab = lab
        self.mask = mask
        self.num = len(self.img)

    def __getitem__(self, idx):
        # global i
        # imgs = []
        imgs = self.img[idx].squeeze(1)
        labs = self.lab[idx].squeeze(0)
        masks = self.mask[idx].squeeze(0)
        # print(imgs.shape,labs.shape)
        # for m in self.modalities:
        # img = img[m,:,:]
        # if self.transform:
        #     img = cv2.resize(img,dsize=self.transform)
        #     labs = cv2.resize(labs,dsize=self.transform)
        # img = self.img_pre(img)
        # labs = self.lab_pre(lab)
        return imgs, labs, masks

    def __len__(self):
        return self.num


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample[0]
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        sample = [image] + [*sample[1:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]

class make_data_3d(Dataset):
    def __init__(self, imgs, plabs, masks, labs):
        self.img = [img.cpu().squeeze() for img in imgs]
        self.plab = [np.squeeze(lab.cpu()) for lab in plabs]
        self.mask = [np.squeeze(mask.cpu()) for mask in masks]
        self.lab = [np.squeeze(lab.cpu()) for lab in labs]
        self.num = len(self.img)
        self.tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((96, 96, 96)),
            # RandomNoise(),
            ToTensor()
        ])

    def __getitem__(self, idx):
        samples = self.img[idx], self.plab[idx], self.mask[idx], self.lab[idx]
        samples = self.tr_transform(samples)
        imgs, plabs, masks, labs = samples
        return imgs, plabs.long(), masks.float(), labs.long()

    def __len__(self):
        return self.num
