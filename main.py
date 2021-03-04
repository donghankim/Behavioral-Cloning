import torch
import numpy as np
import pandas as pd
import cv2
import pdb, os, math
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import *
from model import *
from trainer import *


"""
training is not working...
1. maybe insert all three images to train?
2. create a more powerful model (bring resent)
3. image processing -> crop the images
"""


def main():
    # reading data
    linux_comp = False
    IMG_PATH = 'simulator_data/IMG'
    x_center = []
    x_left = []
    x_right = []
    x_speed = []
    y = []
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

        speed = df.iloc[i]['speed']
        angle = df.iloc[i]['steering_angle']

        # pdb.set_trace()
        x_center.append(center_img)
        x_left.append(left_img)
        x_right.append(right_img)
        x_speed.append(speed)
        y.append(angle)

    # dataset creation
    center_dataset = FrameDataset(x_center, y, True)
    left_dataset = FrameDataset(x_left, y, True)
    right_dataset = FrameDataset(x_right, y, True)
    speed_dataset = LinearDataset(speed, y, True)

    center_train, center_val = random_split(center_dataset, [math.ceil(len(center_dataset)*0.8), math.floor(len(center_dataset)*0.2)])
    left_train, left_val = random_split(left_dataset, [math.ceil(len(left_dataset)*0.8), math.floor(len(left_dataset)*0.2)])
    right_train, right_val = random_split(right_dataset, [math.ceil(len(right_dataset)*0.8), math.floor(len(right_dataset)*0.2)])

    # model + trainer init
    # SimpleNet = Simple(320, 160)
    # Runner = Trainer(SimpleNet)
    Incep = Inception()
    Runner = Trainer(Incep.model)

    Runner.train('base.pth', center_train)
    # Runner.train('base.pth', left_train)
    # Runner.train('base.pth', right_train)




if __name__ == '__main__':
    main()
