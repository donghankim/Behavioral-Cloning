import torch
import numpy as np
import pandas as pd
import cv2
import pdb, os, math, random
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import *
from model import *
from trainer import *
from utils import *

def read_data():
    # change this path if you want to try a different dataset
    IMG_PATH = 'simulator_data/udacity_data/IMG'
    all_imgs = []
    x_center = []
    x_left = []
    x_right = []
    x_speed = []
    y = []
    y_all = []
    # you also need to change this path
    df = pd.read_csv('simulator_data/udacity_data/driving_log.csv')

    print("Reading Data...")
    for i in tqdm(range(len(df))):
        center_token = df.iloc[i]['center'].strip().split("/")
        center_img = cv2.imread(os.path.join(IMG_PATH, center_token[-1])) if os.path.exists(
            os.path.join(IMG_PATH, center_token[-1])) else print(f"no center img for idx:{i}")

        left_token = df.iloc[i]['left'].strip().split("/")
        left_img = cv2.imread(os.path.join(IMG_PATH, center_token[-1])) if os.path.exists(
            os.path.join(IMG_PATH, center_token[-1])) else print(f"no center img for idx:{i}")

        right_token = df.iloc[i]['right'].strip().split("/")
        right_img = cv2.imread(os.path.join(IMG_PATH, center_token[-1])) if os.path.exists(
            os.path.join(IMG_PATH, center_token[-1])) else print(f"no center img for idx:{i}")

        drop_prob = np.random.randn()
        speed = df.iloc[i]['speed']
        angle = round(df.iloc[i]['steering'], 2)

        if angle == 0 and drop_prob < 2.5:
            continue

        all_imgs.append(center_img)
        all_imgs.append(left_img)
        all_imgs.append(right_img)
        y_all.append(round(angle, 2))
        y_all.append(round(angle + 0.22, 2))
        y_all.append(round(angle - 0.22, 2))

        x_center.append(center_img)
        x_left.append(left_img)
        x_right.append(right_img)
        x_speed.append(speed)
        y.append(round(angle, 2))

    return all_imgs, y_all

def main():
    all_imgs, y_all = read_data()
    augmented_imgs, augmented_y = augment_data(all_imgs, y_all)

    # dataset creation
    all_dataset = FrameDataset(augmented_imgs, augmented_y)
    all_train, all_val = random_split(all_dataset, [math.ceil(len(all_dataset)*0.9), math.floor(len(all_dataset)*0.1)])

    # training init
    pdb.set_trace()
    sample_img = all_dataset[0][0]
    SimpleNet = Simple(sample_img.shape[2], sample_img.shape[1])
    Runner = Trainer(SimpleNet)
    Runner.train('base_all.pth', all_dataset)


if __name__ == '__main__':
    main()
