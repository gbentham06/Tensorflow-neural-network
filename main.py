from tensorflow import keras
import numpy as np


# Unpacking Dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# Preprocessing data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


x_test = keras.layers.Flatten()(x_test)
x_train = keras.layers.Flatten()(x_train)

y_train_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_test_hot = keras.utils.to_categorical(y_test, num_classes=10)





print(x_train.shape)
