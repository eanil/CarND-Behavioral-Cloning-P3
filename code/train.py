import glob
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
import cv2
from sklearn.model_selection import train_test_split
from model import *

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
X_train, X_test, y_train, y_test = train_test_split(
    newImagePaths, labels, test_size=0.2)

training_generator = DataGenerator(X_train, y_train, 128)
validation_generator = DataGenerator(X_test, y_test, 128)


print("Training...")

model = dave2Model()

print(model.summary())

# train
model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, validation_split=0.2, shuffle=True, epochs=7)
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    validation_steps=(len(X_test) // 128),
                    use_multiprocessing=True,
                    steps_per_epoch=(len(imagePaths) // 128),
                    workers=2,
                    epochs=1)

model.save('model.h5')
