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

        # converting color code 
        center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        """
        drop_prob = np.random.randn()
        if angle == 0 and drop_prob < 2.5:
            continue
        """

        angle = round(df.iloc[i]['steering'], 2)
        left_angle = round(angle + 0.22, 2)
        right_angle = round(angle - 0.22, 2)
        
        all_imgs.append(center_img)
        all_imgs.append(left_img)
        all_imgs.append(right_img)
        all_imgs.append(cv2.flip(left_img, 1))
        all_imgs.append(cv2.flip(right_img, 1))
    
        y_all.append(angle)
        y_all.append(left_angle)
        y_all.append(right_angle)
        y_all.append(left_angle*-1)
        y_all.append(right_angle*-1)

    return all_imgs, y_all

def main():
    all_imgs, y_all = read_data()
    
    # dataset creation
    all_dataset = FrameDataset(all_imgs, y_all)
    all_train, all_val = random_split(all_dataset, [math.ceil(len(all_dataset)*0.9), math.floor(len(all_dataset)*0.1)])

    # training init
    sample_img = all_dataset[0][0]
    Net = Nvidia(sample_img.shape[2], sample_img.shape[1])
    Runner = Trainer(Net)
    Runner.train('nvidia.pth', all_train)

    # evaluation
    Runner.test('nvidia.pth', all_val)


if __name__ == '__main__':
    main()
