# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:08:29 2020

@author: taoxi
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:27:22 2020

@author: taoxi
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import LeakyReLU
import keras
import numpy as np
from PIL import Image
import argparse
import math
from keras.datasets.mnist import load_data
from keras.utils.vis_utils import plot_model
import keras
batch_size = 64
input_size = 50
epoch_size = 50
def get_mnist_data(number):
    (x1, y1), (x2, y2) = load_data()
    x1 = (x1.astype(np.float32) - 127.5)/127.5
    X =  np.array([x1[i] for i in range(0, len(x1)) if y1[i] == number])
    y = np.array([y1[i] for i in range(0, len(x1)) if y1[i] == number])     
    X = X[:, :, :, None]
    return X, y
  #  x1 = x1[ :, :, :, None]
 #   return x1,y1

def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=50, output_dim=256, activation = 'tanh'))
    #model.add(Activation('tanh'))
    model.add(Dense(128*7*7, activation = 'tanh'))
    model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (5,5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (5, 5), padding='same', activation = 'tanh'))
    #model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3),padding='same',input_shape=(28, 28, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation ='tanh'))
    model.add(Dense(1,activation='sigmoid'))
    return model


def total_model(generater, discriminator):
    model = Sequential()
    model.add(generater)
    discriminator.trainable = False
    model.add(discriminator)
    return model

def get_input_generator():
    return  np.random.uniform(-1, 1, (batch_size, input_size))

def change_into_image(arr):
     image = arr*127.5+127.5
     image = image.reshape((image.shape[0], image.shape[1]))
     return image;

def train(number):   
    X_train, y_train = get_mnist_data(number)
 #   X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    model1 = discriminator_model()
    model2 = generator_model()
    model = total_model(model2, model1)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    model2.compile(loss='binary_crossentropy', optimizer="SGD")
    #model.compile(loss='binary_crossentropy', optimizer="SGD")
    model.compile(loss='binary_crossentropy', optimizer=d_optim)
    model1.trainable = True
    model1.compile(loss='binary_crossentropy', optimizer= d_optim)
    #model1.compile(loss='binary_crossentropy', optimizer="SGD")
    plot_model(model1, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=False)
    plot_model(model2, to_file='generator_plot.png', show_shapes=True, show_layer_names=False)
    for time in range(0, epoch_size):
        print("Current epoch time is", time)
        num = X_train.shape[0]
        for i in range(0,int( num / batch_size)):
            generator_input = get_input_generator()
            fake_image = model2.predict(generator_input, verbose = 0)
            true_image = X_train[i*batch_size : (i+1)*batch_size]
            #print(true_image.shape)
            train_image = np.concatenate((true_image, fake_image)) 
          #  print(train_image.shape)
            train_label = [1 for i in range(0, batch_size)] + [0 for i in range(0, batch_size)]
            discrimator_loss = model1.train_on_batch(train_image, train_label)
            train_image_step2 =  np.random.uniform(-1, 1, (batch_size*3, input_size))
            train_label_step2 = [1 for i in range(0, 3*batch_size)]
            model1.trainable = False
            generator_loss = model.train_on_batch(train_image_step2, train_label_step2)
            model1.trainable = True
            if(i == int(num/batch_size) - 1):
                print("current loss of discrimator is: ", discrimator_loss)
                print("current loss of generator is: ", generator_loss)
                model1.save("discrimator"+ str(number), True)
                model2.save("generator" + str(number), True)
            if(i % 20 ==0):
                print(i)
        model1.save("discrimator"+ str(number), True)
        model2.save("generator" + str(number), True)
def generate(number):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator' + str(number))
    generator_input = get_input_generator()
    images = g.predict(generator_input, verbose=1)
    for i in range(0,batch_size):
        image = change_into_image(images[0])
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize((280, 280))
        image.save(
            "GAN_result/GAN"+str(number)+"_"+str(i)+".png")

for i in range(0,10):
    train(i)
    generate(i)

#train(6)
#generate(6)