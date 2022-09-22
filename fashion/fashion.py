import numpy as np
import pandas as pd
from keras.utils import np_utils
from tensorflow.keras.models import Sequential  # sequence of layers as defined in this library
from tensorflow.keras.layers import Dense  # all neurons are connected

fashion = pd.read_csv("fashion-mnist_train.csv")
fashion = np.array(fashion)
y_train = fashion[:, 0]
y_train = y_train.reshape(-1, 1)
y_train = np_utils.to_categorical(y_train)

x_train = (fashion[:, 1:785])
x_train.astype('float32')
x_train = x_train/255

# 784 -> 397 -> 397 -> 10
network = Sequential()
network.add(Dense(input_shape=(784,), units=397, activation='relu'))
network.add(Dense(units=397, activation='relu'))
network.add(Dense(units=10, activation='softmax'))

network.compile(loss='categorical_crossentropy',
                # for every one correct prediction there could be 10 false predictions so
                # sparse_categorical_crossentropy is recommended
                optimizer='adam',
                metrics=['accuracy'])
history = network.fit(x_train, y_train, batch_size=128, epochs=100)

fashion = pd.read_csv("fashion-mnist_test.csv")
fashion = np.array(fashion)
y_test = fashion[:, 0]
y_test = y_test.reshape(-1, 1)
y_test = np_utils.to_categorical(y_test)
x_test = (fashion[:, 1:785])
x_test.astype('float32')
x_test = x_test/255

accuracy_test = network.evaluate(x_test, y_test)

predictions = network.predict(x_test)
print(accuracy_test)
