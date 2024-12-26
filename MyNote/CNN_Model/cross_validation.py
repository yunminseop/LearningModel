import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import SGD, Adam, RMSprop, Adagrad
from sklearn.model_selection import KFold

matplotlib.use("TkAgg")

# preprocessing
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = y_train.astype(np.float32)/255.0
y_test = y_test.astype(np.float32)/255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# parameters build
n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_output = 10

batch_size = 256
n_epochs = 20
k = 5

# function to build model
def build_model():
    model = Sequential()
    model.add(Dense(units=n_hidden1, activation="relu", input_shape=(n_input,)))
    model.add(Dense(units=n_hidden2, activation="relu"))
    model.add(Dense(units=n_hidden3, activation="relu"))
    model.add(Dense(units=n_hidden4, activation="relu"))
    model.add(Dense(units=n_output, activation="softmax"))
    return model


def cross_validation(opt):
    accuracy = []
    for train_index, val_index in KFold(k).split(x_train):
        xtrain, xval = x_train[train_index], x_train[val_index]
        ytrain, yval = y_train[train_index], y_train[val_index]
        dmlp = build_model()
        dmlp.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        dmlp.fit(xtrain, ytrain, batch_size=batch_size, epochs=n_epochs, verbose=0)
        accuracy.append(dmlp.evaluate(xval,yval,verbose=0)[1])
    return accuracy

acc_sgd = cross_validation(SGD())
acc_adagrad = cross_validation(Adagrad())
acc_rmsprop = cross_validation(RMSprop())
acc_adam = cross_validation(Adam())

print(f"acc of SGD: {np.array(acc_sgd).mean()}")
print(f"acc of Adagrad: {np.array(acc_adagrad).mean()}")
print(f"acc of RMSprop: {np.array(acc_rmsprop).mean()}")
print(f"acc of Adam: {np.array(acc_adam).mean()}")

plt.boxplot([acc_sgd, acc_adagrad, acc_rmsprop, acc_adam], labels=["SGD", "Adagrad", "RMSprop", "Adam"])
plt.grid()
plt.show()