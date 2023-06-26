Source .py files for the masking and mineral characterization algorithm workflow. Data preproccesing, model creation, trainning, prediction, and saving. 

Files:
* distributionnet.py: Classes for hyperspectral mineral characterization's Sampling and Distributional Encoder creation.
* hyperspectrum_models.py: Classes for masking algorithm model creation, trainning, and predition. Also, Classes for hyperspectral mineal characterization to transform hyperspectral data to a common latent variable.
* neuralnet.py: Neural networks construction for hyperspectral mineral characterization with keras library (NNRegresor and NNClassifier). Also,  AutoEncoder and VariationalAutoEncoder classes for Encoders creation.
* plot_functions.py: Plotting functions for the confusion matrix and the predicted mineralogy image.
* pn2tiff.py: Transform a image file from png to tiff.

How to run:
* hyperspectrum_models.py:

