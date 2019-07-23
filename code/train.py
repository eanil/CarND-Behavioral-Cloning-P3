import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

from model import *
from data import *

# read in the data
data = pd.read_csv("../../BehaviorSample/driving_log.csv")

imagePaths = np.hstack((data["right"], data["left"], data["center"]))
labels = np.hstack((data["steering"], data["steering"], data["steering"]))

newImagePaths = []
for imgPath in imagePaths:
    newImgPath = "../../BehaviorSample/IMG/" + imgPath.split('/')[1]
    newImagePaths.append(newImgPath)

data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True)
batch_size = 66

training_generator = DataGeneratorInPlace(data_train, batch_size, 0.2)
validation_generator = DataGeneratorInPlace(data_test, batch_size, 0.2)

print("Training...")

model = dave2Model()

print(model.summary())

plot_model(model, show_shapes=True, to_file='../examples/model.png')

# train
model.compile(loss='mse', optimizer='adam')

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    validation_steps=(len(data_test) // batch_size),
                    use_multiprocessing=True,
                    steps_per_epoch=(len(imagePaths) // batch_size),
                    workers=3,
                    epochs=6)

model.save('model.h5')
