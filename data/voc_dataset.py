import os
import os.path as osp
import numpy as np
import random
import itertools

# import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
from torch.utils import data
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from data.cta import cta_apply
from data.cta.ctaugment import OPS
from torchvision.utils import save_image


class VOCDataSet(data.Dataset):
    def __init__(
        self,
        root,
        list_path,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        scale=True,
        mirror=True,
        ignore_label=255,
    ):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({"img": img_file, "label": label_file, "name": name})

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            image, label = self.generate_scale_label(image, label)
        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0)
            )
            label_pad = cv2.copyMakeBorder(
                label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,),
            )
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(
            img_pad[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w], np.float32,
        )
        label = np.asarray(
            label_pad[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w], np.float32,
        )
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name, index


class VOCGTDataSet(data.Dataset):
    def __init__(
        self,
        root,
        list_path,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        scale=True,
        mirror=True,
        ignore_label=255,
    ):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({"img": img_file, "label": label_file, "name": name})

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]

        attempt = 0
        while attempt < 10:
            if self.scale:
                image, label = self.generate_scale_label(image, label)

            img_h, img_w = label.shape
            pad_h = max(self.crop_h - img_h, 0)
            pad_w = max(self.crop_w - img_w, 0)
            if pad_h > 0 or pad_w > 0:
                attempt += 1
                continue
            else:
                break

        if attempt == 10:
            image = cv2.resize(image, self.crop_size, interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.crop_size, interpolation=cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w = label.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(
            image[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w], np.float32
        )
        label = np.asarray(
            label[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w], np.float32
        )
        image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name


class VOCDataTestSet(data.Dataset):
    def __init__(self, root, list_path, crop_size=(505, 505), mean=(128, 128, 128)):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            self.files.append({"img": img_file})

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        size = image.shape
        name = osp.splitext(osp.basename(datafiles["img"]))[0]
        image = np.asarray(image, np.float32)
        image -= self.mean

        img_h, img_w, _ = image.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0)
            )
        image = image.transpose((2, 0, 1))
        return image, name, size


class VOCCTADataSet(data.Dataset):
    def __init__(
        self,
        root,
        list_path,
        cta,
        max_iters=None,
        crop_size=(321, 321),
        mean=(128, 128, 128),
        # scale=True,
        ignore_label=255,
        # ops_weak=None,
        # ops_strong=None,
        split="train",
    ):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.crop_h, self.crop_w = crop_size
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        # self.scale = scale
        self.files = []
        self.cta = cta
        self.split = split

        for name in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({"img": img_file, "label": label_file, "name": name})

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def deserialize(self, policy):
        ops = []
        bins = []
        for p in policy:
            ops.append(p[0])
            bins.append(str(p[1][0]))
        return (ops, bins)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        label_raw = label

        self.ops_weak = self.cta.policy(probe=False, weak=True)
        self.ops_strong = self.cta.policy(probe=False, weak=False)

        image_weak = cta_apply(Image.fromarray(image), self.ops_weak)
        image_strong = cta_apply(image_weak, self.ops_strong)
        label = cta_apply(Image.fromarray(label), self.ops_weak)
        image_weak, image_strong, label = (
            np.array(image_weak),
            np.array(image_strong),
            np.array(label),
        )
        size = image_weak.shape
        name = datafiles["name"]

        # if self.scale:
        #     image, label = self.generate_scale_label(image, label)
        image_weak = np.asarray(image_weak, np.float32)
        image_strong = np.asarray(image_strong, np.float32)
        image_weak -= self.mean
        image_strong -= self.mean
        img_pad, label_pad = image_weak, label

        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_weak_pad = cv2.copyMakeBorder(
                image_weak, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0)
            )
            img_strong_pad = cv2.copyMakeBorder(
                image_strong, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0.0, 0.0, 0.0)
            )
            label_pad = cv2.copyMakeBorder(
                label, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(self.ignore_label,),
            )
        else:
            img_weak_pad, img_strong_pad, label_pad = image_weak, image_strong, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image_weak = np.asarray(
            img_weak_pad[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w], np.float32,
        )
        image_strong = np.asarray(
            img_strong_pad[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w], np.float32,
        )
        label = np.asarray(
            label_pad[h_off : h_off + self.crop_h, w_off : w_off + self.crop_w], np.float32,
        )

        image_weak = image_weak[:, :, ::-1]  # change to RGB
        image_strong = image_strong[:, :, ::-1]  # change to RGB

        image_weak = image_weak.transpose((2, 0, 1))
        image_strong = image_strong.transpose((2, 0, 1))
        assert (
            image_weak.shape[1:] == image_strong.shape[1:] == label.shape
        ), "image and label must be of the same size"
        # print("label raw", np.unique(label_raw))
        # print("label", np.unique(label))
        # assert np.array_equal(np.unique(label_raw), np.unique(label)), "labels changed"

        # apply augmentations

        # print(
        #     "label raw",
        #     len(np.unique(label_raw)),
        #     " label processed",
        #     len(np.unique(label)),
        #     " weak ops",
        #     self.ops_weak,
        #     name,
        # )
        # to_tensor = transforms.ToTensor()

        # image, image_weak, image_strong, label = (
        #     to_tensor(image),
        #     to_tensor(image_weak),
        #     to_tensor(image_strong),
        #     to_tensor(label)*255,
        # )
        # label = label.squeeze()

        return (
            image_weak.copy(),
            image_strong.copy(),
            label.copy(),
            np.array(size),
            name,
            index,
            self.deserialize(self.ops_weak),
            self.deserialize(self.ops_strong),
        )

        # sample = {
        #     "image": image,
        #     "image_weak": image_weak,
        #     "image_strong": image_strong,
        #     "label_aug": label,
        # }
        # # return sample
        # return (
        #     image_weak,
        #     image_strong,
        #     label,
        #     np.array(size),
        #     name,
        #     index,
        # )


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == "__main__":
    dst = VOCDataSet("./data", is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data in enumerate(trainloader):
        imgs, labels = data
