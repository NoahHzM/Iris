import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD #Stochastic Gradient Descent Optimizer
from keras.datasets import mnist

#load the iris dataset
iris = load_iris()

X = iris.data
encoder = LabelBinarizer()
y = encoder.fit_transform(iris.target)  # We transform to one-hot
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

inputs = Input(shape=(4,))  # We have four inputs
x = Dense(5, activation='sigmoid')(inputs)  # five neurons and sigmoid as activation function
predictions = Dense(3, activation='softmax')(x)  # Output layer with three neurons, one for each class

# This creates a model
model = Model(inputs=inputs, outputs=predictions)
sgd = SGD(lr=0.1)
model.compile(optimizer=sgd,
              loss='mse',  # Mean Squared Error
              metrics=['accuracy'])

print(model.summary())

model.fit(X_train, y_train, epochs=500)  # starts training with 500 epochs

predictions = model.predict(X_test)
for p, l in zip(predictions, y_test):
    print(p, "->", l)


batch_size = 128
num_classes = 10
epochs = 25

# the data, split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)  # 28x28=784
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255  # We normalize to have values between 0 and 1
X_test /= 255

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)  # onehot encoding
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=(784,))
x = Dense(15, activation='sigmoid')(inputs)  # five neurons and sigmoid as activation function
predictions = Dense(10, activation='softmax')(x)  # Output layer with three neurons, one for each class

# This creates a model
model = Model(inputs=inputs, outputs=predictions)
sgd = SGD(lr=0.1)
model.compile(optimizer=sgd,
              loss='mse',  # Mean Squared Error
              metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs)

predictions = model.predict(X_test)
for p, l in zip(predictions, y_test):
    print(p, "->", l)
