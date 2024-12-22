# 2024.12.23. CNN Programming_YunMinSeop
# Source: (오일석, 이진선), "파이썬으로 만드는 인공지능", 한빛아카데미, 2023.01.30, 264-266p
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib
import matplotlib.pyplot as plt

# For setting up back-end which uses GUI
matplotlib.use('TkAgg')

# Read MNIST datasets and reshape into right tensor.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1) # 2D
x_test = x_test.reshape(10000, 28, 28, 1)   # 2D

x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# LeNet-5 Neural Network
CNN = Sequential()
CNN.add(Conv2D(6, (5,5), padding="same", activation="relu", input_shape=(28,28,1))) # C - The model extracts features of the image.
CNN.add(MaxPooling2D(pool_size=(2,2)))                                              # P - It reduces a size of its feature map and main features remain.
CNN.add(Conv2D(16, (5,5), padding="same", activation="relu"))                       # C
CNN.add(MaxPooling2D(pool_size=(2,2)))                                              # P
CNN.add(Conv2D(120, (5,5), padding="same", activation="relu"))                      # C
CNN.add(Flatten())
CNN.add(Dense(84, activation="relu"))                                               # FC - Learning complex non-linear relationship.
CNN.add(Dense(10, activation="softmax"))                                            # FC - Calculating probability distribution.

# compile & fit
CNN.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])
hist_CNN = CNN.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

# evaluate its accuracy
result = CNN.evaluate(x_test, y_test, verbose=0)
print(f"accuracy: {result[1]*100}%")

# plot
plt.plot(hist_CNN.history["accuracy"])
plt.plot(hist_CNN.history["val_accuracy"])
plt.title("CNN model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="best")
plt.grid()
plt.show()

# loss_graph
plt.plot(hist_CNN.history["loss"])
plt.plot(hist_CNN.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="best")
plt.grid()
plt.show()