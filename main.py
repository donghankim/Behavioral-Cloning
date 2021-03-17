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

    print("\nReading Data...")
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

        angle = df.iloc[i]['steering']
        left_angle = angle + 0.22
        right_angle = angle - 0.22
        
        # to randomly get rid of zero angle images
        drop_prob = np.random.randn()
        if angle == 0 and drop_prob < 1.5:
            continue
    
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
    augmented_img, augmented_y = augment_images(all_imgs, y_all)
    
    # dataset creation
    all_dataset = FrameDataset(all_imgs, y_all)
    augmented_dataset = FrameDataset(augmented_img, augmented_y)
    all_train, all_val = random_split(all_dataset, [math.ceil(len(all_dataset)*0.9), math.floor(len(all_dataset)*0.1)])
    aug_train, aug_val = random_split(augmented_dataset, [math.ceil(len(augmented_dataset)*0.9), math.floor(len(augmented_dataset)*0.1)])

    # for debugging purposes
    sample_all = all_dataset[0][0]
    sample_aug = augmented_dataset[0][0]
    
    # training init
    # Net = Nvidia(sample_aug.shape[2], sample_aug.shape[1])
    Net = CarDenseModel()
    Runner = Trainer(Net)
    Runner.train('cardensemodel.pth', augmented_dataset)

    # evaluation
    Runner.test('cardensemodel.pth', aug_val)


if __name__ == '__main__':
    main()
