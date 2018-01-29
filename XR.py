# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:10:29 2018

"""

import cv2
import os
import keras
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


images = glob(os.path.join("images", "*.png"))
labels = pd.read_csv('sample_labels.csv')


    
disease = ["Atelectasis", "Effusion", "Nodule", "Infiltration"]
#dis_dict = {"Atelectasis" : 0, "Cardiomegaly": 1, "Effusion": 2, "Fibrosis": 3, "Pleural_Thickening": 4, "Infiltration": 5, "Mass": 6}
x = [] # images as arrays
y = [] # labels names of diseases or no findings
WIDTH = 128
HEIGHT = 128
    
for img in images:
    base= os.path.basename(img) 
    finding = labels["Finding Labels"][labels["Image Index"] == base].values[0]

    # Read and resize image
    full_size_image = cv2.imread(img)

    # Labels
    if finding in disease:
        y.append(finding)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))

from sklearn.preprocessing import LabelEncoder, LabelBinarizer   
encoder = LabelEncoder()
encoder.fit(y)
dict1 = list(encoder.classes_)
encoded_Y = encoder.transform(y)
y = keras.utils.np_utils.to_categorical(encoded_Y)

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify = y, random_state=0)

from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

K.image_data_format()

img_width, img_height = 128, 128
nb_train_samples = len(X_train)
nb_test_samples = len(X_test)
epochs = 10
batch_size = 32

model = models.Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(np.array(X_train), y_train, batch_size=batch_size)
test_generator = test_datagen.flow(np.array(X_test), y_test, batch_size=batch_size)

model.fit_generator(train_generator, 
                    steps_per_epoch = 26,
                    epochs = 30,
                    validation_data = test_generator,
                    validation_steps = 8)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('SP/il.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result  = model.predict(test_image)
index = np.argmax(result)
print(index)


