# -*- coding: utf-8 -*-
"""
InceptionV3, pretrained model on the imagenet dataset, used for extracting features
from the images
"""

import pathlib
import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing import image
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.applications import inception_v3

# Construct the classes
classes = []
rootdir = '/content/drive/MyDrive/Computer Vision/training/training' # path to the folder with the training data
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        classes.append(file)

# Get the data
images = []
labels = []
label_counter = 0

# Map the image arrays with their label
for cls in classes:
  path = pathlib.Path(os.path.join(rootdir, cls))
  for img in path.glob("*.jpg"):
    img = image.load_img(img, target_size=(255, 255))
    image_array = image.img_to_array(img)
    images.append(image_array)
    labels.append(label_counter)
  label_counter = label_counter + 1

# Convert inputs to numpy arrays
x_train = np.array(images)
y_train = np.array(labels)

# Convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, len(classes))

# Data augmentation
data_augmentation = keras.Sequential(
  [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1)
  ]
)

# Normalize the input
x_train = inception_v3.preprocess_input(x_train)

# Get the pre-trained neural network feature extractor
pretrained_nn = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(255, 255, 3))

# Extract features for each image (all in one pass)
x_train = pretrained_nn.predict(x_train)

# Create a model and add layers
model = Sequential()

model.add(data_augmentation)
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(15, activation='softmax'))

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=50,
    shuffle=True
)

feature_extraction_model = inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(255, 255, 3))

testingdir = '/content/drive/MyDrive/Computer Vision/testing'

# Remove the contents of the run3 file
remove_content = open("run3.txt","w")
remove_content.close()
# Append to the empty run3 file
output = open('run3.txt', 'a')

# Make predictions on the images from the testing folder
for image_name in os.listdir(testingdir):
  image_path = os.path.join(testingdir, image_name)
  loaded_image = keras.preprocessing.image.load_img(image_path, target_size=(255,255))
  array_image = keras.preprocessing.image.img_to_array(loaded_image)
  list_of_images = np.expand_dims(array_image, axis=0)
  images = inception_v3.preprocess_input(list_of_images)
  features = feature_extraction_model.predict(images)
  results = model.predict(features) 
  single_result = results[0]
  most_likely_class_index = int(np.argmax(single_result))
  output.write(image_name + " " + classes[most_likely_class_index] + "\n")

output.close()