# -*- coding: utf-8 -*-
""" 
Transfer learning with VGG16.
"""

import pathlib
import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.preprocessing import image
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from keras.applications import vgg16
import joblib
from tensorflow.python.keras.regularizers import Regularizer

classes = []

rootdir = '/content/drive/MyDrive/Computer Vision/training/training' # path to the training folder
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        classes.append(file)

# get the data
images = []
labels = []
label_counter = 0

for cls in classes:
  path = pathlib.Path(os.path.join(rootdir, cls))
  for img in path.glob("*.jpg"):
    img = image.load_img(img, target_size=(64, 64))
    image_array = image.img_to_array(img)
    images.append(image_array)
    labels.append(label_counter)
  label_counter = label_counter + 1

x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

# Convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, len(classes))
y_test = keras.utils.np_utils.to_categorical(y_test, len(classes))

# Data augmentation
data_augmentation = keras.Sequential(
  [
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1)
  ]
)

x_train = vgg16.preprocess_input(x_train)
pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Extract features for each image (all in one pass)
features_x = pretrained_nn.predict(x_train)

# Save the array of extracted features to a file
joblib.dump(features_x, "x_train.dat")
joblib.dump(y_train, "y_train.dat")

x_train = joblib.load("x_train.dat")
y_train = joblib.load("y_train.dat")

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

# Predicting on all the data
feature_extraction_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

for name in classes:
  c = np.array(list(pathlib.Path(rootdir).glob(name + str('/*'))))
  count = 0
  for im in c:
    try:
      loaded_image = keras.preprocessing.image.load_img(im, target_size=(64,64))
      array_image = keras.preprocessing.image.img_to_array(loaded_image)
      list_of_images = np.expand_dims(array_image, axis=0)
      images = vgg16.preprocess_input(list_of_images)
      features = feature_extraction_model.predict(images)
      results = model.predict(features) 
      single_result = results[0]
      most_likely_class_index = int(np.argmax(single_result))
      if classes[most_likely_class_index] == name:
        count = count + 1
    except:
      continue
  print("For class " + name + " accuracy is " + str(count/len(c)))

# Predicting only on the testing data
count = 0
for i in range(len(x_test)):
  image = x_test[i]
  actual_label = label = np.argmax(y_test[i])
  list_of_images = np.expand_dims(image,axis=0)
  images = vgg16.preprocess_input(list_of_images)
  features = feature_extraction_model.predict(images)
  results = model.predict(features)
  single_result = results[0]
  most_likely_class_index = int(np.argmax(single_result))
  if most_likely_class_index == actual_label:
    count = count + 1

print("Testing accuracy is " + str(count/len(x_test)))
