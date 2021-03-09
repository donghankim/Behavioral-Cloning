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

# flip image also think about modiying the network architecture
def augment_data(images, labels):
    aug_images = []
    aug_labels = []
    for i in range(len(images)):
        img, y = images[i], labels[i]
        flipped_img = cv2.flip(img, 1)
        flipped_y = -1*y

        aug_images.append(img)
        aug_images.append(flipped_img)
        aug_labels.append(y)
        aug_labels.append(flipped_y)

    return aug_images, aug_labels
    










