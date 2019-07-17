import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
import cv2
from sklearn.model_selection import train_test_split

from model import *
from data import *

# read in the data
data = pd.read_csv("../../BehaviorSample/driving_log.csv")

# print some numbers
print("Number of data rows found: ", data.shape[0])
print("Number of image paths found: ", data.shape[0]*3)

imagePaths = np.hstack((data["right"], data["left"], data["center"]))
labels = np.hstack((data["steering"], data["steering"], data["steering"]))

newImagePaths = []
for imgPath in imagePaths:
    newImgPath = "../../BehaviorSample/IMG/" + imgPath.split('/')[1]
    newImagePaths.append(newImgPath)

# X_train, X_test, y_train, y_test = train_test_split(
#     newImagePaths, labels, test_size=0.2)

data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True)

batch_size = 129

training_generator = DataGeneratorInPlace(data_train, batch_size, 0.2)
validation_generator = DataGeneratorInPlace(data_test, batch_size, 0.2)

print("Training...")

model = dave2Model()

print(model.summary())

# train
model.compile(loss='mse', optimizer='adam')

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    validation_steps=(len(data_test) // batch_size),
                    use_multiprocessing=True,
                    steps_per_epoch=(len(imagePaths) // batch_size),
                    workers=6,
                    epochs=1)

model.save('model.h5')
