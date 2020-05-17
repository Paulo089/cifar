import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar100
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from keras.applications import VGG16
import matplotlib.pylab as plt

batch_size = 100
num_classes = 100
epochs = 5

(xt, yt), (xtest, ytest) = cifar100.load_data()

_, filas, columnas, canales = xt.shape

xt = xt.astype('float32')
xtest = xtest.astype('float32')

xt = xt / 255
xtest = xtest / 255

yt = keras.utils.to_categorical(yt, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)



Entradas = Input(shape=(filas, columnas, canales))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(Entradas)
# x=Dropout(0.25)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), name='block1_pool')(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
# x=Dropout(0.25)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), name='block2_pool')(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

modelo = Model(inputs=Entradas, outputs=x)
modelo.summary()
Adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)  # SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

modelo.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam,metrics=['categorical_accuracy'])

history = modelo.fit(xt, yt, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest))

puntuacion = modelo.evaluate(xtest, ytest, verbose=1)

print(puntuacion)

plt.figure(1)
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Precision de Modelo')
plt.ylabel('Precision')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')


plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perdidas del Modelo')
plt.ylabel('Perdidas')
plt.xlabel('Epocas')
plt.legend(['Entrenamiento', 'Test'], loc='upper left')
plt.show()