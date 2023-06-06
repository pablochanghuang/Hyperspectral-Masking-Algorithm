from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.regularizers import l1_l2
from keras import backend as K

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score, accuracy_score

def sampling(args):
    """This function was copied from the Keras documentation found at:
    https://keras.io/examples/variational_autoencoder/
    and referenced on June 14, 2020

    Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VariationalAutoEncoder:
    def __init__(self, n_components=8, activation='relu', output_activation=None):
        self._ncomps = n_components
        self._activation = activation
        self._output_activation = output_activation
        self._encoder = None
        self._decoder = None

    def fit(self, x, **kwargs):
        shp = x.shape[-1]
        inputs = Input(shape=(shp,))
        L = Dense(100, activation=self._activation)(inputs)
        z_mean = Dense(self._ncomps)(L)
        z_log_var = Dense(self._ncomps)(L)
        z = Lambda(sampling, output_shape=(self._ncomps,))([z_mean, z_log_var])

        self._encoder = Model(inputs, [z_mean, z_log_var, z])
        
        latent_inputs = Input(shape=(self._ncomps,))
        L = Dense(100, activation=self._activation)(latent_inputs)
        outputs = Dense(shp, activation=self._output_activation)(L)

        self._decoder = Model(latent_inputs, outputs)

        outputs = self._decoder(self._encoder(inputs)[2])
        vae = Model(inputs, outputs)
        vae.compile(optimizer='adam', loss='mse')
        vae.fit(x=x, y=x, **kwargs)
    
    def transform(self, x):
        return self._encoder.predict(x)[0]

    def inverse_transform(self, encoded_x):
        return self._decoder.predict(encoded_x)

class AutoEncoder:
    def __init__(self, n_components=8, activation='relu', output_activation=None):
        self._ncomps = n_components
        self._activation = activation
        self._output_activation = output_activation
        self._encoder = None
        self._decoder = None

    def fit(self, x, **kwargs):
        shp = x.shape[-1]
        inputs = Input(shape=(shp,))
        L = Dense(100, activation=self._activation)(inputs)
        encoded = Dense(self._ncomps)(L)

        self._encoder = Model(inputs, encoded)

        encoded_inputs = Input(shape=(self._ncomps,))
        L = Dense(100, activation=self._activation)(encoded_inputs)
        outputs = Dense(shp, activation=self._output_activation)(L)

        self._decoder = Model(encoded_inputs, outputs)

        outputs = self._decoder(self._encoder(inputs))
        autoencoder = Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(x=x, y=x, **kwargs)

    def transform(self, x):
        return self._encoder.predict(x)

    def inverse_transform(self, encoded_x):
        return self._decoder.predict(encoded_x)        

class NNRegressor:
    '''Constructs a neural network regressor that interfaces well with sklearn functions while
    using tensorflow/keras backend. The model has a variety of simple custimization options that
    are typically avaiable with keras.
        - loss : supply the loss function as a string or a function
        - hidden_layers : a list of hidden layer sizes
        - activation : an activation function to apply to all the hidden layers (string or callable)
        - activations : a list of activation functions of the same length as hidden_layers
        - l1 : l1-norm penalty applied at the final layer
        - l2 : l2-norm penalty applied at the final layer
    '''
    def __init__(self, loss='mse', hidden_layers=(100,), activation='relu', activations=None, l1=0, l2=1e-4):
        self._mdl = None
        self._loss = loss
        self._reg = l1_l2(l1, l2)
        if (activations is not None) and (len(activations) != len(hidden_layers)):
            raise(Exception("When using keyword argument 'activations', the activations must appear as a list of length equal to that of 'hidden_layers'."))
        self._lyrs = hidden_layers
        if activations is not None:
            self._acts = activations
        else:
            self._acts = [activation for i in range(len(hidden_layers))]

    def fit(self, x, y, **kwargs):
        '''fit the model.
            - additional arguments are passed to the keral Model.fit function
        '''
        inputs = Input(shape=(x.shape[-1],))

        L = Dense(self._lyrs[0], activation=self._acts[0])(inputs)
        for i, units in enumerate(self._lyrs, 1):
            L = Dense(units, activation=self._acts[i])(L)

        outputs = Dense(y.shape(-1), kernel_regularizer=self._reg)(L)

        self._mdl = Model(inputs, outputs)
        self._mdl.compile(optimizer='adam', loss=self._loss)
        self._mdl.fit(x=x, y=y, **kwargs)

    def predict(self, x):
        return self._mdl.predict(x)

    def score(self, x, y):
        return r2_score(y, self.predict(x))

class NNClassifier:
    '''Constructs a neural network classifier that interfaces well with sklearn functions while
    using tensorflow/keras backend. The model has a variety of simple custimization options that
    are typically avaiable with keras.
        - loss : supply the loss function as a string or a function
        - hidden_layers : a list of hidden layer sizes
        - activation : an activation function to apply to all the hidden layers (string or callable)
        - activations : a list of activation functions of the same length as hidden_layers, takes
            priority over 'activation' when not set to None.
        - l1 : l1-norm penalty applied at the final layer
        - l2 : l2-norm penalty applied at the final layer
    '''
    def __init__(self, loss='categorical_crossentropy', hidden_layers=(100,), activation='relu', activations=None, l1=0, l2=1e-4):
        self._lb = LabelBinarizer()
        self._mdl = None
        self._loss = loss
        self._reg = l1_l2(l1, l2)
        if (activations is not None) and (len(activations) != len(hidden_layers)):
            raise(Exception("When using keyword argument 'activations', the activations must appear as a list of length equal to that of 'hidden_layers'."))
        self._lyrs = hidden_layers
        if activations is not None:
            self._acts = activations
        else:
            self._acts = [activation for i in range(len(hidden_layers))]
    
    def fit(self, x, y, **kwargs):
        '''fit the model.
            - additional arguments are passed to the keral Model.fit function
        '''
        c = self._lb.fit_transform(y)

        inputs = Input(shape=(x.shape[-1],))

        L = Dense(self._lyrs[0], activation=self._acts[0])(inputs)
        for i, units in enumerate(self._lyrs, 1):
            L = Dense(units, activation=self._acts[i])(L)

        outputs = Dense(c.shape(-1), activation='softmax', kernel_regularizer=self._reg)(L)

        self._mdl = Model(inputs, outputs)
        self._mdl.compile(optimizer='adam', loss=self._loss)
        self._mdl.fit(x=x, y=c, **kwargs)

    def predict_proba(self, x):
        return self._mdl.predict(x)

    def predict(self, x):
        return self._lb.inverse_transform(self.predict_proba(x))

    def score(self, x, y):
        return accuracy_score(y, self.predict(x))
