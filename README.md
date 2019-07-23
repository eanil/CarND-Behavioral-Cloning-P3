# **Behavioral Cloning** 

This project implements and trains a CNN to drive a car around a track using the Udacity Autonomous Driving Simulator.

The network and the approach I used here are the same as the original NVIDIA paper. See https://devblogs.nvidia.com/deep-learning-self-driving-cars/

The network, training code, and auxiliary code to prepare training and validation data in batches were inplemented using Keras.

---

[//]: # (Image References)

[image1]: ./examples/network.png "Model Visualization"
[image2]: ./examples/img1.png "Original frame"
[image3]: ./examples/imgCropped.png "Cropped and Resized Image"
[image4]: ./examples/imgFlipped.png "Flipped Image"
[image5]: ./examples/histo.png "Steering angle histogram"

---
### Files

All code files can be found inside the *code* directory.

code/train.py contains the code to train the network.

code/data.py contains the data generator to augment and prepare the batches for training and validation.

code/model.py contains the network.

Additionally, I also changed the original drive.py file so that I can use cropped and resized images during inference.

### Training and Validation Data

I drove around the track and then back to prepare the training data. Even then, as the histogram shows the data was unbalanced. The straight frames overwhelm the data. 

![alt text][image5]

To fix this issue, I did two things:

* Use the right and left camera images in addition to the center images. For left camera images I added a correction factor to the steering angles to the center camera steering angles. For the right images, I subtracted the same factor.
* Flipping the images to obtain more training data.

Here is an original frame:

![alt text][image2]

Here is the same frame after cropping and resizing:

![alt text][image3]

Finally, after flipping the cropped image:

![alt text][image4]

We will talk more on how and when to augment the data below.

### Model Architecture

I used the network proposed in the original NVIDIA paper. Here is what the network architecture looks like.

![alt text][image1]

The input to the network is 3-channel RGB images. Next the network uses 5 convolutional layers. The first 3 layers use 5x5 kernels with 2 pixel strides. The next 2 layers use 3x3 kernels with no stride. Finally, the data is flattened and passed through 3 fully convolutional layers. At each step RELU activation function is applied. The output is a single number representing the steering angle. 

### Training Strategy and Implementation

For training in batches efficiently, I implemented a data generator using keras.utils.Sequence.

See code/data.py for the augmentation and data generator code.

The benefit of taking this approach is that we can augment the data on the fly as batches are constructed. Hence, we don't need to save extra images on disk or use extra memory for the augmented images beyond what is necessary for the batch. With a generator we can also construct the batches on CPU an use GPU for training.

Implementing a Sequence is pretty straightforwad. We only need to derive a new class from keras.utils.Sequence and implement 3 functions: __init__, __getitem__, and __len__. 

__getitem__ is the important function that constructs the batches and returns the data we will use. I chose to augment the data inside this function. Everytime this function is called for a new batch, it picks a number of rows from the original data, adds the left, and right images, flips the set and adjusts the steering angles. Finally, it crops and resizes the images and returns the set.

I shuffled the data when feeding it to the network for training and validation. 

The data was also divided into 80/20 training and validation. Both the training and validation sets were fed into the network using the data generator we implemented above after shuffling.

I used Adam optimizer to train the network. 

After trying a couple of values, I selected the batch size 66. 

### Results

The approach worked quite well. In the first try I was able to make the car drive around the track without going out of the track at all. I also experimented with changing the velocity of the car in drive.py. The video I'm including in this repo was shot with 20mph. See video.mp4 for results. 

We could have tried to make it a little more robust by adding more swerves to the training data.
