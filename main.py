import torch
import numpy as np
import pandas as pd
import cv2
import pdb, os, math
from torch.utils.data import random_split
from tqdm import tqdm

from dataset import *

def main():

    # reading data
    IMG_PATH = 'simulator_data/IMG'
    x_center = []
    x_left = []
    x_right = []
    x_speed = []
    y = []
    df = pd.read_csv('simulator_data/driving_log.csv')

    print("Reading Data...")
    for i in tqdm(range(len(df))):
        center_img = cv2.imread(df.iloc[i]['center_path'].strip()) if os.path.exists(df.iloc[i]['center_path'].strip()) else print(f"no center img for idx:{i}")
        left_img = cv2.imread(df.iloc[i]['left_path'].strip()) if os.path.exists(df.iloc[i]['left_path'].strip()) else print(f"no left img for idx:{i}")
        right_img = cv2.imread(df.iloc[i]['right_path'].strip()) if os.path.exists(df.iloc[i]['right_path'].strip()) else print(f"no right img for idx:{i}")
        speed = df.iloc[i]['speed']
        angle = df.iloc[i]['steering_angle']

        x_center.append(center_img)
        x_left.append(left_img)
        x_right.append(right_img)
        x_speed.append(speed)
        y.append(angle)

    # dataset creation
    center_dataset = FrameDataset(x_center, y)
    left_dataset = FrameDataset(x_left, y)
    right_dataset = FrameDataset(x_right, y)
    speed_dataset = LinearDataset(speed, y)

    center_test, center_val = random_split(center_dataset, [math.ceil(len(center_dataset)*0.8), math.floor(len(center_dataset)*0.2)])
    left_test, left_val = random_split(left_dataset, [math.ceil(len(left_dataset)*0.8), math.floor(len(left_dataset)*0.2)])
    right_test, right_val = random_split(right_dataset, [math.ceil(len(right_dataset)*0.8), math.floor(len(right_dataset)*0.2)])



if __name__ == '__main__':
    main()
