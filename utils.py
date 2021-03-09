import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os, pdb


"""
I think I have to use subplots

fig, ax = plt.subplots(figsize=(5, 3))
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(x1, y1*10000)
ax.set_xlabel('time [s]')
ax.set_ylabel('Damped oscillation [V]', labelpad=18)

plt.show()
"""

# view histogram of training data
def view_hist(y_values):
    values = {}
    for i in range(len(y_values)):
        if y_values[i] not in values.keys():
            values[y_values[i]] = 1
        else:
            values[y_values[i]] += 1


    total_imgs = sum(list(values.values()))
    plt.figure(figsize = (12, 5))
    plt.bar(values.keys(), values.values(), align = 'edge', width=0.3)
    plt.xlim(-1.5, 1.5)
    plt.show()
    pdb.set_trace()

def augment_data(dataset):
    pass







