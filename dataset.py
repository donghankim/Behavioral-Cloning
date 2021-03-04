import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage import io
import pdb, os


class FrameDataset(Dataset):
    def __init__(self, X, Y, downloaded = False):
        self.X = X
        self.labels = Y
        if downloaded:
            self.transfrom = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                    0.229, 0.224, 0.225]),
            ])

    def show_img(self, img, denormalize=True):
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
        if denormalize:
            img = inv_normalize(img)

        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.show()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img = self.X[index]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_label = torch.tensor(float(self.labels[index]))

        if self.transform:
            img_rgb = self.transform(img_rgb)

        return (img_rgb, y_label)


class LinearDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.labels = Y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.X[index], self.labels[index])


