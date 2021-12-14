# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

import os
import glob
from typing import Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm
from urllib.request import urlretrieve
import shutil
import pytorch_lightning as pl
from torchvision import transforms


class ImageDrivingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str = "data/driving_dataset", height=256, width=455, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        # Data Augmentation strategy
        self.transform = transforms.Compose([
            transforms.CenterCrop((height, width)),
            # transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def prepare_data(self) -> None:
        # Download the data needed
        ImageDrivingDataset(self.data_dir, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        full_set = ImageDrivingDataset(data_dir=self.data_dir, transform=self.transform)

        train_len = len(full_set) * 7 / 10  # 70% of dataset is for training
        val_len = len(full_set) - train_len  # 30% of dataset is for validation
        self.train_set, self.val_set = random_split(full_set, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)


class VideoDrivingDataset(Dataset):
    def __init__(self, data_dir="data/driving_dataset", sequence=16, download=True, transform=None) -> None:
        self.dataset = ImageDrivingDataset(data_dir=data_dir, download=download, transform=transform)
        self.sequence = sequence

    def __len__(self):
        return len(self.dataset) // self.sequence

    def __getitem__(self, index):
        image_index = self.sequence * index

        frames = angles = []
        for i in range(image_index, image_index + self.sequence):
            frame, angle = self.dataset[i]
            frames.append(frame)
            angles.append(angle)

        return torch.tensor(frames), torch.tensor(angles)


class ImageDrivingDataset(Dataset):
    """Images of SullyChen's "07/01/2018 Driving Dataset" from https://github.com/SullyChen/driving-datasets"""

    DATASET_URL = "https://nextcloud.univ-lille.fr/index.php/s/2CKyZzoLBPN4qLF/download/07012018.zip"

    def __init__(self, data_dir="data/driving_dataset", download=True, transform=transforms.ToTensor()) -> None:
        self.data_dir = data_dir

        # download and extract if not exist
        if not os.path.exists(self.data_dir):
            if not download:
                raise RuntimeError(
                    'Dataset does not exist. You have to download it first by enabling download=True in argument.')
            else:
                print('Downloading dataset...')
                self._download_and_extract()

        # load list of data
        self.data = self._load_data()  # format : List[Tuple(frame_path, angle)]

        self.transform = transform

    def _load_data(self):
        filepath = os.path.join(self.data_dir, 'data.txt')
        file = open(filepath, 'r')
        lines = file.readlines()

        data = []
        for line in lines:
            # format of lines is : "filename.jpg angle,year-mm-dd hr:min:sec:millise"
            frame_filename, info = line.split()

            frame_path = os.path.join(self.data_dir, 'data', frame_filename)
            angle = float(info.split(sep=',')[0])

            data.append((frame_path, angle))

    def _download_and_extract(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        filepath = os.path.join(self.data_dir, "archive.zip")
        download_url(
            ImageDrivingDataset.DATASET_URL,
            filepath
        )
        extract_archive(filepath)
        os.remove(filepath)

        print('\n##################\nDataset installed successfully.\n################')

    def __getitem__(self, index: int):
        frame_path, angle = self.data[index]

        # read the RGB image from the frame path
        image = Image.open(frame_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, angle

    def __len__(self):
        return len(self.data)


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Dataset already exists on the disk. Skipping download.")
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)


class PilotNetDataset(Dataset):
    """PilotNet dataset class that preserver temporal continuity. Returns
    images and ground truth values when the object is indexed.
    Parameters
    ----------
    path : str
        Path of the dataset folder. If the folder does not exists, the folder
        is created and the dataset is downloaded and extracted to the folder.
        Defaults to '../data'.
    sequence : int
        Length of temporal sequence to preserve. Default is 16.
    transform : lambda
        Transformation function to be applied to the input image.
        Defaults to None.
    train : bool
        Flag to indicate training or testing set. Defaults to True.
    visualize : bool
        If true, the train/test split is ignored and the temporal sequence of
        the data is preserved. Defaults to False.
    Examples
    --------
    >>> dataset = PilotNetDataset()
    >>> images, gts = dataeset[0]
    >>> num_samples = len(dataset)
    """

    def __init__(
        self, path='data', sequence=16,
        train=True, visualize=False, transform=None, extract=True
    ):
        self.path = path + '/driving_dataset/'

        dataset_link = 'https://drive.google.com/file/d/'\
            '0B-KJCaaF7elleG1RbzVPZWV4Tlk/view?usp=sharing'
        download_msg = f'''Please download dataset form \n{dataset_link}')
        and copy driving_dataset.zip to {path}/
        Note: create the folder if it does not exist.'''.replace(' ' * 8, '')

        # check if dataset is available in path. If not download it
        if len(glob.glob(self.path)) == 0:
            if extract is True:
                if os.path.exists(path + '/driving_dataset.zip'):
                    print('Extracting data (this may take a while) ...')
                    os.system(
                        f'unzip {path}/driving_dataset.zip -d {path} '
                        f'>> {path}/unzip.log'
                    )
                    print('Extraction complete.')
                else:
                    print(f'Could not find {path + "/driving_dataset.zip"}.')
                    raise Exception(download_msg)
            else:
                print('Dataset does not exist. set extract=True')
                if not os.path.exists(path + '/driving_dataset.zip'):
                    raise Exception(download_msg)

        with open(self.path + '/data.txt', 'r') as data:
            all_samples = [line.split() for line in data]

        self.samples = all_samples

        if visualize is True:
            inds = np.arange(len(all_samples) // sequence)
        else:
            inds = np.random.RandomState(
                seed=42
            ).permutation(len(all_samples) // sequence)
        if train is True:
            self.ind_map = inds[
                :int(len(all_samples) / sequence * 0.8)
            ] * sequence
        else:
            self.ind_map = inds[
                -int(len(all_samples) / sequence * 0.2):
            ] * sequence

        self.sequence = sequence
        self.transform = transform

    def __getitem__(self, index: int):
        images = []
        gts = []
        for i in range(self.sequence):
            path, gt = self.samples[self.ind_map[index] + i]
            if np.abs(float(gt)) < 1e-5 and i != 0 and i != len(self.samples) - 1:
                gt = 0.5 * (  # removing dataset anomalities
                    float(self.samples[self.ind_map[index] + i - 1][1]) +
                    float(self.samples[self.ind_map[index] + i + 1][1])
                )
            image = Image.open(self.path + path)
            gt_val = float(gt) * np.pi / 180
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
            gts.append(torch.tensor(gt_val, dtype=image.dtype))

        images = torch.stack(images, dim=3)
        gts = torch.stack(gts, dim=0)

        return images, gts

    def __len__(self):
        return len(self.ind_map)
