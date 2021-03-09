import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
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
    ax.set_title(f"Training Image Distribution ({total_imgs} images)")
    ax.set_xlabel("Angle Value")
    plt.show()

    #pdb.set_trace()

def augment_data(dataset):
    pass










