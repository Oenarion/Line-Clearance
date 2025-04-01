import os
from glob import glob
from pathlib import Path
import shutil
import numpy as np
import csv
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torchvision.datasets as datasets
from torchvision.datasets import CIFAR10
import random


class Dataset_maker(torch.utils.data.Dataset):
    def __init__(self, root, category, config, is_train=0):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),  
                transforms.ToTensor(), # Scales data into [0,1] 
                transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
            ]
        )
        self.config = config
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((config.data.image_size, config.data.image_size)),
                transforms.ToTensor(), # Scales data into [0,1] 
            ]
        )
        #train data
        if is_train == 0:
            if category:
                image_files = glob(
                    os.path.join(root, category, "train", "good", "*.bmp")
                )
            else:
                image_files = glob(
                    os.path.join(root, "train", "good", "*.bmp")
                )
            random.seed(42)
            train_data_len = int(len(image_files) * 0.8)
            self.image_files = random.sample(image_files, train_data_len)
        #validation data
        elif is_train == 1:
            if category:
                image_files = glob(
                    os.path.join(root, category, "train", "good", "*.bmp")
                )
            else:
                image_files = glob(
                    os.path.join(root, "train", "good", "*.bmp")
                )
            random.seed(42)
            train_data_len = int(len(image_files) * 0.8)
            self.image_files = list(set(image_files) - set(random.sample(image_files, train_data_len)))
        else:
            if category:
                self.image_files = glob(os.path.join(root, category, "test", "*", "*.bmp"))
            else:
                self.image_files = glob(os.path.join(root, "test", "*", "*.bmp"))
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = self.image_transform(image)
        # print(image_file.replace("\\test\\","\\ground_truth\\"))
        if(image.shape[0] == 1):
            image = image.expand(3, self.config.data.image_size, self.config.data.image_size)
        if self.is_train == 0:
            label = 'good'
            return image, label
        else:
            if self.config.data.mask:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    if self.config.data.name == 'AVDataset':
                        target = Image.open(
                            image_file.replace("\\test\\","\\ground_truth\\")).convert('L')
                    else:
                        target = Image.open(
                            image_file.replace("\\test\\","\\ground_truth\\")).convert('L')
                    target = self.mask_transform(target)
                    label = 'defective'
            else:
                if os.path.dirname(image_file).endswith("good"):
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'good'
                else :
                    target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                    label = 'defective'

            return image, target, label

    def __len__(self):
        return len(self.image_files)
