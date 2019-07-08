
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
from keras.utils import Sequence
import cv2
import numpy as np


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        '''
        x_set: Filenames of the images we want to use.
        y_set: Target values
        batch_size: 
        '''
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        'Number of batches per epoch'
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        'Returns the batch.'
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([cv2.imread(file_name) for file_name in batch_x]), np.array(batch_y)


def dave2Model(loss='mse', optimizer='adam'):
    model = Sequential()

    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(160, 320, 3)))
    model.add(Conv2D(filters=24, kernel_size=5,
                     strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5,
                     strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5,
                     strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3,
                     strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3,
                     strides=(1, 1), activation='relu'))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer)

    return model
