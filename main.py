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


def main():
    # reading data
    linux_comp = True
    IMG_PATH = 'simulator_data/IMG'
    all_imgs = []
    x_center = []
    x_left = []
    x_right = []
    x_speed = []
    y = []
    y_all = []
    df = pd.read_csv('simulator_data/driving_log.csv')

    print("Reading Data...")
    for i in tqdm(range(len(df))):
        if linux_comp:
            center_token = df.iloc[i]['center_path'].strip().split("/")
            center_img = cv2.imread(os.path.join(IMG_PATH, center_token[-1])) if os.path.exists(os.path.join(IMG_PATH, center_token[-1])) else print(f"no center img for idx:{i}")

            left_token = df.iloc[i]['left_path'].strip().split("/")
            left_img = cv2.imread(os.path.join(IMG_PATH, center_token[-1])) if os.path.exists(os.path.join(IMG_PATH, center_token[-1])) else print(f"no center img for idx:{i}")

            right_token = df.iloc[i]['right_path'].strip().split("/")
            right_img = cv2.imread(os.path.join(IMG_PATH, center_token[-1])) if os.path.exists(os.path.join(IMG_PATH, center_token[-1])) else print(f"no center img for idx:{i}")
        else:
            center_img = cv2.imread(df.iloc[i]['center_path'].strip()) if os.path.exists(df.iloc[i]['center_path'].strip()) else print(f"no center img for idx:{i}")
            left_img = cv2.imread(df.iloc[i]['left_path'].strip()) if os.path.exists(df.iloc[i]['left_path'].strip()) else print(f"no left img for idx:{i}")
            right_img = cv2.imread(df.iloc[i]['right_path'].strip()) if os.path.exists(df.iloc[i]['right_path'].strip()) else print(f"no right img for idx:{i}")

        drop_prob = np.random.randn()
        speed = df.iloc[i]['speed']
        angle = round(df.iloc[i]['steering_angle'], 2)

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

    # data augmentation
    all_imgs, y_all = augment_data(all_imgs, y_all)

    # dataset creation
    all_dataset = FrameDataset(all_imgs, y_all)
    speed_dataset = LinearDataset(speed, y) 
    all_train, all_val = random_split(all_dataset, [math.ceil(len(all_dataset)*0.9), math.floor(len(all_dataset)*0.1)])
   
    # training init
    sample_img = all_dataset[0][0]
    SimpleNet = Simple(sample_img.shape[2], sample_img.shape[1])
    Runner = Trainer(SimpleNet)
    Runner.train('base_all.pth', all_dataset)


if __name__ == '__main__':
    main()
