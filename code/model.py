
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout
import tensorflow as tf

def dave2Model(loss='mse', optimizer='adam'):
    model = Sequential()

    model.add(Lambda(lambda x:  (x / 127.5) - 1., input_shape=(66, 200, 3)))
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
