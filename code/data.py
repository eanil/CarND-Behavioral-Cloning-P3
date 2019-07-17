from keras.utils import Sequence
import cv2
import numpy as np
import tensorflow as tf


def bgr2rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def flipimg(image):
    return cv2.flip(image, 1)


def cropimg(image):
    cropped = image[60:130, :]
    return cropped


def resize(image, shape=(160, 70)):
    return cv2.resize(image, shape)


def crop_and_resize(image):
    cropped = cropimg(image)
    resized = resize(cropped)
    return resized


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

        return np.array([cv2.imread(file_name) for file_name in batch_x]), \
            np.array(batch_y)


class DataGeneratorInPlace(Sequence):
    def __init__(self, data, batch_size, steering_correction):
        '''
        data: original dataset from csv
        batch_size: target batch size.
        steering_correction: correction to apply to left and right images
        in degrees.

        The generator will actually pick less than the batchsize
        directly from the data and augment the rest.

        For example, if batch_size = 128, it will pick 128/8=16 rows from the data.
        32 rows will give 16 * 3 = 48 data points. We will augment an additional
        80 data points by randomly picking from the original rows.

        This assumes that the data has the following columns:
        ['center', 'left', 'right', 'steering']
        '''
        # assert (batch_size % 8 == 0), 'Choose a batch size divisible by 8'

        self.data = data
        self.batch_size = batch_size
        self.batch_rows = int(batch_size / 3)  # / 8
        self.angle_correction = 0.2

    def __len__(self):
        'Number of batches per epoch'
        return int(np.ceil(len(self.data) / float(self.batch_rows)))

    def __getitem__(self, idx):
        'Returns the batch'
        batch = self.data[idx * self.batch_rows:(idx + 1) * self.batch_rows]

        # We will make this many augmented images
        # because each row contributes 3 data points.
        aug_size = self.batch_size - self.batch_rows * 3

        # imagePaths = np.array(batch["center"])
        imagePaths = np.hstack(
            (batch['center'], batch['right'], batch['left'])).tolist()
        center_angles = np.array(batch["steering"]).astype(float)
        angles = np.hstack((center_angles,
                            center_angles - self.angle_correction,
                            center_angles + self.angle_correction)).tolist()

        def fullPath(imgPath): return "../../BehaviorSample/IMG/" + \
            imgPath.split('/')[1]

        images = [cv2.imread(fullPath(file_name))
                  for file_name in imagePaths]

        # flip the images

        return np.array(images), np.array(angles)
