import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
import cv2
from sklearn.model_selection import train_test_split

# read in the data
data = pd.read_csv("../../BehaviorSample/driving_log.csv")

# print some numbers
print("Number of data rows found: ", data.shape[0])
print("Number of image paths found: ", data.shape[0]*3)


for idx, d in data.iterrows():
    # do nothing with center

    # flip left, flip steering and add to the end of the list

    # flip right, flip steering and add to the end of the list

