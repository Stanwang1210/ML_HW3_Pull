import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace


print("Loading mean")
my_mean = np.load("cnn_structure/pop_mean.npy")
print("mean loaded")
print("Loading std")
my_std = np.load("cnn_structure/pop_std0.npy")
print("std loaded")

class Classifier_no_norm(nn.Module):
    def __init__(self):
        super(Classifier_no_norm, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.3),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Dropout(0.3),
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
            nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

class Classifier_norm(nn.Module):
    def __init__(self):
        super(Classifier_norm, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.03),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.03),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.2),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.03),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.2),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.03),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Dropout(0.2),
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.03),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
            nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.03),
            nn.MaxPool2d(2, 2, 0),       # [512, 2, 2]
            nn.Dropout(0.2),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.03),
            nn.MaxPool2d(2, 2, 0),       # [512, 1, 1]
            nn.Dropout(0.2),

            
        )
        self.fc = nn.Sequential(
            # nn.Linear(512, 1024),
            # #nn.Dropout(0.1),
            # nn.LeakyReLU(0.03),
            # nn.Linear(1024, 512),
            # #nn.Dropout(0.1),
            # nn.LeakyReLU(0.03),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.03),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.03),
            nn.Linear(128, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)



class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'

        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
            transforms.RandomRotation(15), #隨機旋轉圖片
            transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
            # transforms.Normalize(my_mean,my_std),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize(my_mean,my_std),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform

    # 這個 FoodDataset 繼承了 pytorch 的 Dataset class
    # 而 __len__ 和 __getitem__ 是定義一個 pytorch dataset 時一定要 implement 的兩個 methods
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # 這個 method 並不是 pytorch dataset 必要，只是方便未來我們想要指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

class FoodDataset_norm(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'

        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
            transforms.RandomRotation(15), #隨機旋轉圖片
            transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
            transforms.Normalize(my_mean,my_std),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(my_mean,my_std),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform

    # 這個 FoodDataset 繼承了 pytorch 的 Dataset class
    # 而 __len__ 和 __getitem__ 是定義一個 pytorch dataset 時一定要 implement 的兩個 methods
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # 這個 method 並不是 pytorch dataset 必要，只是方便未來我們想要指定「取哪幾張圖片」出來當作一個 batch 來 visualize
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)
