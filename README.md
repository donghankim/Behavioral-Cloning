# Behavioral Cloning
Using a provided simulator, a neural network is used to clone the driving behavior of a given user.

## Overview
Using a simulator, we can record the steering angle and speed of a moving car. The car in the simulator has three cameras: left, center and right. This setup is similar to the setup used by the paper published by Nvidia, [End to End Learning for Self-Driving Cars.](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) The basic idea is to map the image features from the three cameras to an appropriate steering angle. This information is sent to the simulator frame by frame.

## Installation & Run
The simulator I used is provided by Udacity (click [here](https://github.com/udacity/self-driving-car-sim) to install the simulator). You can record your own data from the simulator, or you can download the provided dataset from Udacity.

With the simulator installed, you also have to install some dependencies (this was a pain to do). I have provided a .yml file (conda environment) so that you dont have to go through the hassle of installing each package one-by-one.
```python
conda env create -f env.yml
```
I have uploaded a few model weights which you can use straight away. In order to test your model run the drive.py file with a model weight of your choosing:
```python
python drive.py model_name.pth
```
The rest of this repository contains code for creating and training the neural network. I have tried to make it as readable and usable as possible.

## Data Processing
After reading in your data, you can see that the range of steering angles are not balanced. There are way to many 0.0 angle images, which will cause the network to output 0.0 for every single frame. Thus you first have to find a way to balance your dataset. I decided to randomly discard 85% of all 0.0 angles and images from the dataset. This might not be ideal, but it does balance the dataset. In addition, we have to include an offset angle to the left and right camera images. I didn't know how much offset I should I add, but after some googling I decided to offset the left camera angle by +0.22 and the right camera angle by -0.22.

## Data Augmentation
Another issue with the dataset is the limited number of examples. Of course, you could technically record the simulator for hours and hours trying to collect as much data as you can, but thats just not efficient. Therefore, I decided to augment my data to include shiffted, brightness altered, shadowed and flipped images. Im not going to go in detail regarding the procedure to augment images since there is a ton of resources out there on the internet.

## Model Architecture & Training
I have include 6 different network architecures in the model.py file. Model Navostha, CarModel, CarDenseModel and CarSimpleModel are from other users on github who have attemtped this project before (I included their network for comaparisson to my network). The network I created is derived from the Nvidia paper mentioned at the top of this README file.

-- Image of network architecure

Initially, I did try to normalize my images at input, but this proved to produce worse results. I also tried to add dropout in several layers, but again this did not produce good results. Therefore, I did not add any normalizations when training my network. However, by looking at the final result, the network does seem to generalize well on the simulator. Furthermore, using the RELU activation function also proved to produce undesirable results. It's interesting how changing the activation function to ELU produces much better results. I also trained my network for 25 epochs, making sure the network was not overfitting to the training data. 





