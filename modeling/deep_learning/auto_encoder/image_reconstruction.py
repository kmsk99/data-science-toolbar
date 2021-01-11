# load the libraries
from PIL import Image
from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from numpy import argmax, array_equal
import matplotlib.pyplot as plt
from keras.models import Model
from imgaug import augmenters
from random import randint
import pandas as pd
import numpy as np

# Dataset Prepration

# read dataset
train = pd.read_csv("../input/fashion-mnist_train.csv")
train_x = train[list(train.columns)[1:]].values
train_y = train['label'].values

# normalize and reshape the predictors
train_x = train_x / 255

# create train and validation datasets
train_x, val_x, train_y, val_y = train_test_split(
    train_x, train_y, test_size=0.2)

# reshape the inputs
train_x = train_x.reshape(-1, 784)
val_x = val_x.reshape(-1, 784)

# Create Autoencoder architecture

# input layer
input_layer = Input(shape=(784,))

# encoding architecture
encode_layer1 = Dense(1500, activation='relu')(input_layer)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)
encode_layer3 = Dense(500, activation='relu')(encode_layer2)

# latent view
latent_view = Dense(10, activation='sigmoid')(encode_layer3)

# decoding architecture
decode_layer1 = Dense(500, activation='relu')(latent_view)
decode_layer2 = Dense(1000, activation='relu')(decode_layer1)
decode_layer3 = Dense(1500, activation='relu')(decode_layer2)

# output layer
output_layer = Dense(784)(decode_layer3)

model = Model(input_layer, output_layer)

model.summary()

# early stopping callback

model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
model.fit(train_x, train_x, epochs=20, batch_size=2048,
          validation_data=(val_x, val_x), callbacks=[early_stopping])

preds = model.predict(val_x)

# Inputs : Actual Images

f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(val_x[i].reshape(28, 28))
plt.show()

# Predicted : Autoencoder Output

f, ax = plt.subplots(1, 5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(preds[i].reshape(28, 28))
plt.show()
