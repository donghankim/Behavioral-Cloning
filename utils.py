import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import os, pdb


# view histogram of training data
def view_hist(y_values):
    values = {}
    for i in range(len(y_values)):
        if y_values[i] not in values.keys():
            values[y_values[i]] = 1
        else:
            values[y_values[i]] += 1


    total_imgs = sum(list(values.values()))
    fig, ax = plt.subplots(figsize = (16, 8))
    fig.subplots_adjust(bottom=0.15, left=0.2)
    ax.bar(values.keys(), values.values(), width = 0.005)
    ax.set_title(f"Augmented Training Image Distribution ({total_imgs} images)")
    ax.set_xlabel("Angle Value")
    plt.show()
    pdb.set_trace()

"""
def augment_data(images, labels):
    aug_images = []
    aug_labels = []
    for i in range(len(images)):
        img, y = images[i], labels[i]
        

        aug_images.append(img)
        aug_images.append(flipped_img)
        aug_labels.append(y)
        aug_labels.append(flipped_y)

    return aug_images, aug_labels
"""

# for model creation
def calc_out_size(current_w, current_h, k_size, pooling, stride):
    w = int((current_w - k_size + 2*pooling)/stride + 1)
    h = int((current_h - k_size + 2*pooling)/stride + 1)
    return w, h

def calc_pool_size(current_w, current_h, k_size, stride):
    w = int((current_w - k_size)/stride + 1)
    h = int((current_h - k_size)/stride + 1)
    return w, h







