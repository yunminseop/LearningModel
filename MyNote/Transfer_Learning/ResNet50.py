import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import os

train_folder = "CUB200/train"
test_folder = "CUB200/test"

class_reduce = 0.1
number_of_class = int(len(os.listdir(train_folder))*class_reduce)

x_train, y_train = [], []
for i, class_name in enumerate(os.listdir(train_folder)):
    if i < number_of_class:
        for fname in os.listdir(train_folder+'/'+class_name):
            img = image.load_img(train_folder+'/'+class_name+ '/' + fname, target_size=(224, 224))
            if len(img.getbands())!=3:
                print("warning, invalid frame exist", class_name, fname)
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x_train.append(x)
            y_train.append(i)

x_test, y_test = [], []
for i, class_name in enumerate(os.listdir(test_folder)):
    if i < number_of_class:
        for fname in os.listdir(test_folder+'/'+class_name):
            img=image.load_img(test_folder+'/'+class_name+'/'+fname, target_size=(224, 224))
            if len(img.getbands())!=3:
                print("warning, invalid frame exist", class_name, fname)
                continue
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x_test.append(x)
            y_test.append(i)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

y_train = tf.keras.utils.to_categorical(y_train, number_of_class)
y_test = tf.keras.utils.to_categorical(y_test, number_of_class)

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
cnn = Sequential()
cnn.add(base_model)
cnn.add(Flatten)
cnn.add(Dense(1024, activation="relu"))
cnn.add(Dense(number_of_class, activation="softmax"))

cnn.compile(loss="categorical_crossentropy", optimizer=Adam(0.00002), metrics=["accuracy"])
hist = cnn.fit(x_train, y_train, batch_size=16, epochs=10, validation_data=(x_test, y_test), verbose=1)
res = cnn.evaluate(x_test, y_test, verbose=0)
print("Accuracy:", res[1]*100)