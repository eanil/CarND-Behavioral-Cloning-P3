from keras.utils import Sequence
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg

def crop(image):
    return image[60:130, :]

def resize(image, shape=(200, 66)):
    return cv2.resize(image, shape)

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
        assert (batch_size % 6 == 0), 'Choose a batch size divisible by 6'

        self.data = data
        self.batch_size = batch_size
        self.batch_rows = int(batch_size / 6)
        self.angle_correction = 0.2

    def __len__(self):
        'Number of batches per epoch'
        return int(np.ceil(len(self.data) / float(self.batch_rows)))

    def __getitem__(self, idx):
        'Returns the batch'
        batch = self.data[idx * self.batch_rows:(idx + 1) * self.batch_rows]

        imagePaths = np.hstack(
            (batch['center'], batch['right'], batch['left'])).tolist()
        center_angles = np.array(batch["steering"]).astype(float)
        angles = np.hstack((center_angles,
                            center_angles - self.angle_correction,
                            center_angles + self.angle_correction)).tolist()

        def fullPath(imgPath): return "../../BehaviorSample/IMG/" + \
            imgPath.split('/')[1]

        images = [resize(crop(mpimg.imread(fullPath(file_name))))
                  for file_name in imagePaths]

        # Flip the images
        imagesFlipped = [np.fliplr(img) for img in images]

        images = np.concatenate((images, imagesFlipped))
        angles = np.concatenate((angles, -1.0 * np.array(angles)))

        return np.array(images), np.array(angles)
