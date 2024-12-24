# 2024.12.24. CNN Programming_YunMinSeop
# Source: (오일석, 이진선), "파이썬으로 만드는 인공지능", 한빛아카데미, 2023.01.30, 274-276p

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

matplotlib.use("TkAgg")

# Preprocessing
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype(np.float32)/255.0
x_test = x_test.astype(np.float32)/255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# Building a CNN Model
CNN = Sequential() # C-C-P-Dropout-C-C-P-Dropout-F-FC-Dropout-FC
CNN.add(Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3))) # C
CNN.add(Conv2D(32, (3,3), activation="relu")) # C
CNN.add(MaxPooling2D(pool_size=(2,2))) # P
CNN.add(Dropout(0.25)) # Dropout
CNN.add(Conv2D(64, (3,3), activation="relu")) # C
CNN.add(Conv2D(64, (3,3), activation="relu")) # C
CNN.add(MaxPooling2D(pool_size=(2,2))) # P
CNN.add(Dropout(0.25)) # Dropout
CNN.add(Flatten()) # F
CNN.add(Dense(512, activation="relu")) # FC
CNN.add(Dropout(0.5)) # Dropout
CNN.add(Dense(10, activation="softmax")) # FC

# Model compile & fit
CNN.compile(loss="categorical_crossentropy", optimizer=Adam(), metrics=['accuracy'])

# ** First try - batch_size=32 / epochs=30
# ** Second try - batch_size=64 / epochs=30
# ** Third try - batch_size=128 / epochs=30
# ** Fourth try - batch_size=128 / epochs=60
# ** Fifth try - batch_size=128 / epochs=90
# ** Sixth try - batch_size=128 / epochs=120
hist_CNN = CNN.fit(x_train, y_train, batch_size=128, epochs=120, validation_data=(x_test, y_test), verbose=2)

# Evaluation
result = CNN.evaluate(x_test, y_test, verbose=0)
print(f"acc: {result[1]*100}%")

# Plot(accuracy)
plt.plot(hist_CNN.history["accuracy"])
plt.plot(hist_CNN.history["val_accuracy"])
plt.title("Accuracy_CNN, batch = 128/ epochs = 120")
plt.legend(["Train", "Validation"], loc="best")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.grid()
plt.show()

# Plot(loss)
plt.plot(hist_CNN.history["loss"])
plt.plot(hist_CNN.history["val_loss"])
plt.title("Loss_CNN, batch = 128/ epochs = 120")
plt.legend(["Train", "Validation"], loc="best")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.grid()
plt.show()