# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:19:48 2020

@author: taoxi
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 01:24:53 2020

@author: taoxi
"""
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Reshape,Lambda, Input
from keras.layers import Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras import backend as K
import keras
import numpy as np
from PIL import Image
import argparse
import math
from keras.datasets.mnist import load_data
from keras.utils.vis_utils import plot_model
from keras import objectives
from scipy.stats import norm
from keras.utils.vis_utils import plot_model
import keras
batch_size = 20
def change_into_image(arr):
     image = arr*255
     image = image.reshape((image.shape[0], image.shape[1]))
     return image;
def get_mnist_data(number):
    (x1, y1), (x2, y2) = load_data()
    x1 = x1.astype(np.float32)/255
    X =  np.array([x1[i] for i in range(0, len(x1)) if y1[i] == number])
    y = np.array([y1[i] for i in range(0, len(x1)) if y1[i] == number])     
    X = X[:, :, :, None]
    return X, y



def encoder_model():
    first = Input(shape=(28,28,1))
    encoder = Conv2D(64, (3, 3),padding='same')(first)
    encoder = LeakyReLU(alpha=0.2)(encoder)
    encoder = Dropout(0.4)(encoder)
    encoder = MaxPooling2D(pool_size= (2,2))(encoder)
    encoder = Conv2D(128, (5, 5),padding='same')(encoder)
    encoder = LeakyReLU(alpha=0.2)(encoder)
    encoder = Dropout(0.4)(encoder)
    encoder = MaxPooling2D(pool_size= (2,2))(encoder)
    encoder = Flatten()(encoder)
    encoder = Dense(1024, activation='relu')(encoder)
    mu = Dense(2, activation='relu')(encoder)
    log_var = Dense(2, activation='relu')(encoder)
    encoder = Model(first, mu)
    return encoder, mu, log_var,first


def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, 2), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps
encoder, mu, log_var,first= encoder_model()
z = Lambda(sampling, output_shape=(2,))([mu, log_var])
decoder= Dense(1024)(z)
decoder = Dense(128*7*7, activation = 'tanh')(decoder)
decoder = BatchNormalization()(decoder)
decoder = Reshape((7, 7, 128), input_shape=(128*7*7,))(decoder)
decoder = Conv2DTranspose(128, (5,5), strides=(2,2), padding='same')(decoder)
decoder = LeakyReLU(alpha=0.2)(decoder)
decoder = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(decoder)
decoder = LeakyReLU(alpha=0.2)(decoder)
decoder_output = Conv2D(1, (5, 5), padding='same', activation = 'sigmoid')(decoder)


for number in range(0, 10):
    X,y = get_mnist_data(number)
    upper = int(X.shape[0]/batch_size) * batch_size
    reconstruction_loss = objectives.binary_crossentropy(K.flatten(first), K.flatten(decoder_output)) * X.shape[0]
    kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
    vae_loss = reconstruction_loss + kl_loss
    
    # build model
    vae = Model(first, decoder_output)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
 #   vae.summary()
    vae.fit(X[:upper],
           shuffle=True,
           epochs=20,
           batch_size=batch_size)
    vae.save("VAE"+str(number), True)
    generator_input = np.random.uniform(-1, 1, (batch_size, 28,28,1))
    images = vae.predict(generator_input, verbose=1)
    for i in range(0, 10):
       # print(images.shape)
        image = change_into_image(images[i])
        image = Image.fromarray(image.astype(np.uint8))
        image = image.resize((280, 280))
        image.save("./VAE_result/VAE_generated_image_"+str(number)+"_"+str(i)+".png")
       #steps_per_epoch=int(X.shape[0]/64))
           #validation_data=(x_te, None), verbose=1)
plot_model(vae, to_file='VAE_plot.png', show_shapes=True, show_layer_names=False)