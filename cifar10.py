import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from keras.applications import VGG16

batch_size = 100
num_classes = 10
epochs = 30

(xt, yt), (xtest, ytest) = cifar10.load_data()

_, filas, columnas, canales = xt.shape

xt = xt.astype('float32')
xtest = xtest.astype('float32')

xt = xt / 255
xtest = xtest / 255

yt = keras.utils.to_categorical(yt, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)

Basica = 0
if (Basica == 1):
    Entradas = Input(shape=(filas, columnas, canales))
    x = Conv2D(64, kernel_size=(3, 3), activation='relu')(Entradas)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

if (Basica == 0):
    Entradas = Input(shape=(filas, columnas, canales))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(Entradas)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Flatten()(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

modelo = Model(inputs=Entradas, outputs=x)
descenso_gradiente_estocastico = Adam(lr=0.001, beta_1=0.9, beta_2=0.9)  # SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
modelo.compile(loss=keras.losses.categorical_crossentropy, optimizer=descenso_gradiente_estocastico,
               metrics=['categorical_accuracy'])

modelo.fit(xt, yt, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(xtest, ytest))

puntuacion = modelo.evaluate(xtest, ytest, verbose=1)

print(puntuacion)