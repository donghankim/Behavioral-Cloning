import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
from tqdm import tqdm
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

def shift_img(image, steer):
    """
    randomly shift image horizontally
    add proper steering angle to each image
    """
    max_shift = 55
    max_ang = 0.14  # ang_per_pixel = 0.0025

    rows, cols, _ = image.shape

    random_x = np.random.randint(-max_shift, max_shift + 1)
    dst_steer = steer + (random_x / max_shift) * max_ang
    if abs(dst_steer) > 1:
        dst_steer = -1 if (dst_steer < 0) else 1

    mat = np.float32([[1, 0, random_x], [0, 1, 0]])
    dst_img = cv2.warpAffine(image, mat, (cols, rows))
    return dst_img, dst_steer

def brightness_img(image):
    """
    randomly change brightness by converting Y value
    """
    br_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    coin = np.random.randint(2)
    if coin == 0:
        random_bright = 0.2 + np.random.uniform(0.2, 0.6)
        br_img[:, :, 2] = br_img[:, :, 2] * random_bright
    br_img = cv2.cvtColor(br_img, cv2.COLOR_HSV2RGB)
    return br_img

def generate_shadow(image, min_alpha=0.5, max_alpha = 0.75):
    """generate random shadow in random region"""
    top_x, bottom_x = np.random.randint(0, image.shape[1], 2)
    coin = np.random.randint(2)
    rows, cols, _ = image.shape
    shadow_img = image.copy()
    if coin == 0:
        rand = np.random.randint(2)
        vertices = np.array([[(50, 65), (45, 0), (145, 0), (150, 65)]], dtype=np.int32)
        if rand == 0:
            vertices = np.array([[top_x, 0], [0, 0], [0, rows], [bottom_x, rows]], dtype=np.int32)
        elif rand == 1:
            vertices = np.array([[top_x, 0], [cols, 0], [cols, rows], [bottom_x, rows]], dtype=np.int32)
        mask = image.copy()
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
        cv2.fillPoly(mask, [vertices], ignore_mask_color)
        rand_alpha = np.random.uniform(min_alpha, max_alpha)
        cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadow_img)

    return shadow_img

def flip_img(image, steering):
    """ randomly flip image to gain right turn data (track1 is biaed in left turn) """
    flip_image = image.copy()
    flip_steering = steering
    num = np.random.randint(2)
    if num == 0:
        flip_image, flip_steering = cv2.flip(image, 1), -steering
    return flip_image, flip_steering

def augment_images(images, lables):
    augmented_images = []
    augmented_y = []

    print("\nAugmenting Images...")
    for i in tqdm(range(len(images))):
        img, y = images[i], lables[i]
        shifted_img, shifted_y = shift_img(img, y)
        bright = brightness_img(img)
        shadow_img = generate_shadow(img)
        flipped_img, flipped_y = flip_img(img, y)

        augmented_images.append(shifted_img)
        augmented_images.append(bright)
        augmented_images.append(shadow_img)
        augmented_images.append(flipped_img)

        augmented_y.append(shifted_y)
        augmented_y.append(y)
        augmented_y.append(y)
        augmented_y.append(flipped_y)
    
    return augmented_images, augmented_y

        
# for model definition
def calc_out_size(current_w, current_h, k_size, pooling, stride):
    w = int((current_w - k_size + 2*pooling)/stride + 1)
    h = int((current_h - k_size + 2*pooling)/stride + 1)
    return w, h

def calc_pool_size(current_w, current_h, k_size, stride):
    w = int((current_w - k_size)/stride + 1)
    h = int((current_h - k_size)/stride + 1)
    return w, h







