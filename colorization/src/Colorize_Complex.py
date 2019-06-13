import argparse
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.datasets import cifar10
from keras.layers import (Activation, Add, BatchNormalization, Conv2D,
                          Conv2DTranspose, Input, Lambda, Softmax,
                          ZeroPadding2D)
from keras.models import Sequential
from keras.preprocessing.image import (ImageDataGenerator, img_to_array,
                                       load_img)
from skimage import img_as_ubyte
from skimage.color import lab2rgb, rgb2lab

import cv2


def build_model(batch_size=16, h=32, w=32, nb_classes= 313):
  input_shape = (h, w, 1) # Channels last
  
  def output_shape(input_shape):
      return (batch_size, h, w, nb_classes + 1)

  def reshape_softmax(x):
      x = K.reshape(x, (batch_size * h * w, nb_classes))
      x = K.softmax(x)
      # Add a zero column so that x has the same dimension as the target (313 classes + 1 weight)
      xc = K.zeros((batch_size * h * w, 1))
      x = K.concatenate([x, xc], axis=1)
      # Reshape back to (batch_size, h, w, nb_classes + 1) to satisfy keras' shape checks
      x = K.reshape(x, (batch_size, h, w, nb_classes + 1))
      return x

  model = Sequential()
  model.add(Conv2D(64, kernel_size=3, name='conv1', input_shape=input_shape, activation='relu', padding='same'))
  
  for i in range(0,3):
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))

    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation("relu"))

  model.add(Conv2D(nb_classes, (1, 1), name="convFinal", padding="same"))

  model.add(Lambda(lambda z: reshape_softmax(z), output_shape=output_shape, name="ReshapeSoftmax"))
  
  return model


def colorizex(modelBatchx, x_black, q_ab, T=0.38):
  X_colorized = modelBatchx.predict(x_black)[:, :, :, :-1]
  
  batch_size, h, w, _ = x_black.shape
  
  X_colorized = X_colorized.reshape((batch_size * h * w, 313))

  #Adjust temperature
  X_colorized = np.exp(np.log(X_colorized) / T)
  X_colorized = X_colorized / np.sum(X_colorized, 1)[:, np.newaxis]

  q_a = q_ab[:, 0].reshape((1, 313))
  q_b = q_ab[:, 1].reshape((1, 313))
  X_a = np.sum(X_colorized * q_a, 1).reshape((batch_size, h, w, 1))
  X_b = np.sum(X_colorized * q_b, 1).reshape((batch_size, h, w, 1))
  
  X_colorized = np.concatenate((x_black * 100, X_a, X_b), axis=3)
  return X_colorized


def colorize_one(file, image, q_ab, T):
  modelBatch1 = build_model(batch_size=1)
  modelBatch1.load_weights(file, by_name=True)

  l,a,b = cv2.split(rgb2lab(image))
  test = []
  test.append(np.reshape(l/100., (32,32,1)))
  test = np.asarray(test)
  colorized_image = colorizex(modelBatch1, test, q_ab, T)
  return img_as_ubyte(lab2rgb(colorized_image[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Colorization')
    parser.add_argument('--model', default="../data/complex.h5",
                        type=str, help='Keras model to load')
    parser.add_argument('--img', default="../data/frog10.png",
                        type=str, help='Image to colorize')
    parser.add_argument('--pts', default="../data/pts_in_hull.npy",
                        type=str, help='Path to pts_in_hull.npy')

    args = parser.parse_args()

    model_path = args.model
    img_path = args.img
    path_pts_in_hull = args.pts

    img = img_to_array(load_img(img_path)).astype('uint8')
    q_ab = np.load(path_pts_in_hull)

    rgb_col = colorize_one(model_path, img, q_ab, 0.1)
    plt.imshow(img)
    plt.show()
    plt.imshow(rgb_col)
    plt.show()
