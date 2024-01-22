# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Load and preprocess the images
# Replace the path with your own image directory
image_dir = "/content/bro"
image_paths = glob.glob(image_dir + "/*.jpg")
images = []
for path in image_paths:
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, (64, 64))
  image = (image - 127.5) / 127.5 # Normalize to [-1, 1]
  images.append(image)
images = tf.stack(images)

#Instructions
As you can see, all you need is to take the name of the
location and then change the number of epochs as you can write here

# Set some hyperparameters
epochs = 230
batch_size = 32
buffer_size = 60000
seed = tf.random.normal([16, 100])
