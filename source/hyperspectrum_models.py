import numpy as np

from scipy.interpolate import interp1d

import pandas as pd

from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler, minmax_scale
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from source.neuralnet import VariationalAutoEncoder, AutoEncoder

from keras import backend as K
from keras.callbacks import EarlyStopping

class Masker:
    '''This class produces a callable which inputs a hyperspectral image (crate type)
    and outputs a binary mask indicating sections of the image labeled useful (1,True)
    and background (0,False).
        - model_pipeline : an object with .predict member function that classifies
            hyperspectral curves.
        - wavelength_range : tuple of length 2, range of usable wavelengths if relevant.
    '''
    def __init__(self, model_pipeline, wavelength_range=None):
        '''
        Description: Creates a model pipeline used to mask the image to differentiate useful
        data to background in an specific wavelenght range. 
        Inputs:
            - model_pipeline : Pipeline object with .predict member function that classifies
            useful core data from background.
            - wavelength_range : tuple of length 2, range of usable wavelengths if relevant 
            (Mostlly SWIR range in Masking Algorithm ).
        Outputs:
            None, saved class attributes.
        '''
        self._model = model_pipeline
        self._idx = wavelength_range
        
    def __call__(self, image):
        '''
        Description: Produces a mask with the model_pipeline and .predict for the image input.
        Inputs:
            - image : Spectral image object, hyperspectral image opened and load with envi (spectral).
        Outputs:
            - yhat : Mask binary output dividing data (1) form background (0).
        '''
        
        if self._idx is not None:
            image = image[:,:,self._idx]
        yhat = self._model.predict(
            image.reshape(-1, image.shape[-1])
        )
        yhat = yhat.reshape(image.shape[:2])
        return yhat

class LatentTransformer:
    '''This class transforms hyperspectral data from thin sections and crate images to a common
    latent variable usable for classifying both data types despite only having automated mineralogy
    labels for the thin sections.
        - overlap_range : tuple of length 2, lower and upper bounds for the wavelength range
            common to both data types.
        - n_components : dimension of latent variable.
        - model_type : "SVD", "AE", or "VAE", encoder type for latent model. "SVD" performs a linear
            dimensionality reduction by a truncated singular value decomposition, while
            "AE" performs a nonlinear dimensionality reduction using a neural network
            auto-encoder. "VAE" uses a variational auto-encoder.
    '''
    def __init__(self, overlap_range, n_components=8, model_type="SVD", **kwargs):
        if (len(overlap_range) != 2):
            raise(Exception(f"overlap_range must be of length 2 (len was {len(overlap_range)})."))
        
        if overlap_range[0] >= overlap_range[1]:
            raise(Exception(f"overlap_range (={overlap_range}) must define a valid and non-empty interval."))

        self._overlap = overlap_range ## wavelength range (min, max,) encompassed in both thin-section and crate

        act = lambda x: K.log(K.square(K.relu(x))+1)
        if model_type == "SVD":
            self._encoder = TruncatedSVD(n_components=n_components, **kwargs)
        elif model_type == "AE":
            self._encoder = AutoEncoder(n_components=n_components, activation=act, **kwargs)
        elif model_type == "VAE":
            self._encoder = VariationalAutoEncoder(n_components=n_components, activation=act, **kwargs)
        else:
            raise(Exception(f"model type (={model_type}) must either be 'SVD' or 'AE' or 'VAE'"))
        self._type = model_type

        self._wl = None ## wavelengths of interest that intersect all data
        self._wl_c = None ## wavelengths for crate data
        self._idx_t = None ## index of wavelengths for thin-sec data
        self._c_scaler = StandardScaler() ## standard scaler model for crate data
        self._t_scaler = StandardScaler() ## standard scaler model for thin-sec data
        
    def fit(self, wl_c, hypr_c, wl_t, hypr_t):
        '''fits the two data types to a common latent variable
            - wl_c : shape (N,), wavelengths for crate data
            - hypr_c : shape (M,N), hyperspectral curves from crate data
            - wl_t : shape(K,) wavelengths for thin-section data
            - hypr_t : shape(L,K), hyprspectral curves from thin-section data
        '''
        assert len(hypr_c.shape) == 2 # input data in matrix form
        assert len(hypr_t.shape) == 2 # input data in matrix form

        ## determine range in thin-sections
        self._idx_t = np.logical_and(self._overlap[0] <= wl_t, wl_t <= self._overlap[1])
        self._wl = wl_t[self._idx_t]

        ## interpolate crate data on useful points
        self._wl_c = wl_c
        intrp = interp1d(wl_c, hypr_c, copy=False)
        Hc = intrp(self._wl)

        ## scale the thin section data
        H = self._t_scaler.fit_transform(hypr_t[:, self._idx_t])

        ## scale the crate data
        self._c_scaler.fit(Hc)

        ## encoder model
        if self._type == "SVD":
            self._encoder.fit(H)
        else:
            fit_monitor = EarlyStopping(monitor='loss', min_delta=1e-4, patience=5)
            self._encoder.fit(H, epochs=200, batch_size=200, callbacks=[fit_monitor])
        
    def transform_c(self, hypr_c):
        '''evaluates the transformation for new hyperspectral curves of crate data type.
            - hypr_t : shape (M,N) where N = len(wl_c) from fit, hyprspectral curves
                from thin-section data.
        '''
        intrp = interp1d(self._wl_c, hypr_c, copy=False)
        Hc = intrp(self._wl)
        Hc = self._c_scaler.transform(Hc)
        Hc = self._encoder.transform(Hc)
        return Hc
    
    def transform_t(self, hypr_t):
        '''evaluates the transformation for new hyperspectral curve of thin-section type.
            - hypr_c : shape (L,K) where K = len(wl_t) from fit, hyperspectral curves
                from crate data.
        '''
        H = hypr_t[:, self._idx_t]
        H = self._t_scaler.transform(H)
        H = self._encoder.transform(H)
        return H
        
class LatentTransformer2:
    def __init__(self, overlap_range, n_components=8, model_type="SVD", **kwargs):
        if (len(overlap_range) != 2):
            raise(Exception(f"overlap_range must be of length 2 (len was {len(overlap_range)})."))
        
        if overlap_range[0] >= overlap_range[1]:
            raise(Exception(f"overlap_range (={overlap_range}) must define a valid and non-empty interval."))

        self._overlap = overlap_range ## wavelength range (min, max,) encompassed in both thin-section and crate

        act = lambda x: K.log(K.square(K.relu(x))+1)
        if model_type == "SVD":
            self._encoder = TruncatedSVD(n_components=n_components, **kwargs)
        elif model_type == "AE":
            self._encoder = AutoEncoder(n_components=n_components, activation=act, **kwargs)
        elif model_type == "VAE":
            self._encoder = VariationalAutoEncoder(n_components=n_components, activation=act, **kwargs)
        else:
            raise(Exception(f"model type (={model_type}) must either be 'SVD' or 'AE' or 'VAE'"))
        self._type = model_type

        self._wl = None ## wavelengths of interest that intersect all data
        self._wl_c = None ## wavelengths for crate data
        self._idx_t = None ## index of wavelengths for thin-sec data

    def fit(self, wl_c, hypr_c, wl_t, hypr_t):
        '''fits the two data types to a common latent variable
            - wl_c : shape (N,), wavelengths for crate data
            - hypr_c : shape (M,N), hyperspectral curves from crate data
            - wl_t : shape(K,) wavelengths for thin-section data
            - hypr_t : shape(L,K), hyprspectral curves from thin-section data
        '''
        assert len(hypr_c.shape) == 2 # input data in matrix form
        assert len(hypr_t.shape) == 2 # input data in matrix form

        ## determine range in thin-sections
        self._idx_t = np.logical_and(self._overlap[0] <= wl_t, wl_t <= self._overlap[1])
        self._wl = wl_t[self._idx_t]

        ## interpolate crate data on useful points
        self._wl_c = wl_c

        ## scale the thin section data
        H = scale(hypr_t[:, self._idx_t])

        ## encoder model
        if self._type == "SVD":
            self._encoder.fit(H)
        else:
            fit_monitor = EarlyStopping(monitor='loss', min_delta=1e-4, patience=5)
            self._encoder.fit(H, epochs=200, batch_size=200, callbacks=[fit_monitor])
    
    def transform_c(self, hypr_c):
        '''evaluates the transformation for new hyperspectral curves of crate data type.
            - hypr_t : shape (M,N) where N = len(wl_c) from fit, hyprspectral curves
                from thin-section data.
        '''
        intrp = interp1d(self._wl_c, hypr_c, copy=False)
        Hc = intrp(self._wl)
        Hc = scale(Hc)
        Hc = self._encoder.transform(Hc)
        return Hc
    
    def transform_t(self, hypr_t):
        '''evaluates the transformation for new hyperspectral curve of thin-section type.
            - hypr_c : shape (L,K) where K = len(wl_t) from fit, hyperspectral curves
                from crate data.
        '''
        H = hypr_t[:, self._idx_t]
        H = scale(H)
        H = self._encoder.transform(H)
        return H
