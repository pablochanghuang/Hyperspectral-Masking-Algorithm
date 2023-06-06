from keras.layers import Lambda, Input, Dense, Layer
from keras.models import Model
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.regularizers import l1_l2
from keras import backend as K

import tensorflow as tf

from sklearn.neighbors import NearestNeighbors

import pandas as pd
import numpy as np

class samplingLayer(Layer):
    def __init__(self, sampler):
        self._sampler = sampler

    def call(self, args):
        return self._sampler(args)

class Sampler:
    def __init__(self, N=100):
        self._k = N
        self._nn = NearestNeighbors()
        self._X = None

    def fit(self, x):
        self._nn.fit(x)
        self._X = x
    
    def __call__(self, z):
        idx = self._nn.kneighbors(z, self._k, return_distance=False)
        i = np.random.randint(self._k)
        idx = idx[i]
        return self._X[idx,:]

class DistributionalEncoder:
    def __init__(self, k=100, activation='relu', output_activation=None, loss='mse'):
        self._activation = activation
        self._output_activation = output_activation
        self._loss = loss
        self._encoder = None
        self._decoder = None
        self._sampler = Sampler(k)

    def fit(self, x, y, **kwargs):
        n_components = y.shape[-1]
        self._sampler.fit(y)

        shp = x.shape[-1]
        inputs = Input(shape=(shp,))
        L = Dense(100, activation=self._activation)(inputs)
        z_estimate = Dense(n_components)(L)
        z = samplingLayer(self._sampler)(z_estimate)

        self._encoder = Model(inputs, [z_estimate, z])

        latent_inputs = Input(shape=(n_components,))
        L = Dense(100, activation=self._activation)(latent_inputs)
        outputs = Dense(shp, activation=self._output_activation)(L)

        self._decoder = Model(latent_inputs, outputs)
        outputs = self._decoder(self._encoder(inputs)[1])
        vae = Model(inputs, outputs)
        vae.compile(optimizer='adam', loss=self._loss)
        vae.fit(x=x, y=x, **kwargs)

    def transform(self, x):
        return self._encoder.predict(x)[0]

    def inverse_transform(self, encoded_x):
        return self._decoder.predict(encoded_x)
