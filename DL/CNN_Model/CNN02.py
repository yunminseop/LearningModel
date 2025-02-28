# 2024.12.23. CNN Programming_YunMinSeop
# Source: (오일석, 이진선), "파이썬으로 만드는 인공지능", 한빛아카데미, 2023.01.30, 267-269p

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

CNN = Sequential()
CNN.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
CNN.add(Conv2D(64, (3, 3), activation="relu"))
CNN.add(MaxPooling2D(pool_size=(2,2)))
CNN.add(Dropout(0.25))
CNN.add(Flatten())
CNN.add(Dense(128, activation="relu"))
CNN.add(Dropout(0.5))
CNN.add(Dense(10, activation="softmax"))

CNN.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])
hist_CNN = CNN.fit(x_train, y_train, batch_size=128, epochs=12, validation_data=(x_test, y_test), verbose=2)

result = CNN.evaluate(x_test, y_test, verbose=0)
print(f"accuracy: {result[1]*100}%")

plt.plot(hist_CNN.history["accuracy"])
plt.plot(hist_CNN.history["val_accuracy"])
plt.title("CNN Model accuracy")
plt.ylabel("Accuarcy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="best")
plt.grid()
plt.show()

plt.plot(hist_CNN.history["loss"])
plt.plot(hist_CNN.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="best")
plt.grid()
plt.show()