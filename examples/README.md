Notebooks .ipynb files that properly import code from source and show data examples of how to use the hyperspectral masking and mineral characterization algorithms. 

Files:
* CNN_masker.ipynb: Creates a CNN model (with an encoder and decoder of 13 convolutional layers each) for masking to an image size of (530,340,411).
* Distribution Model.ipynb: Create Distribution model to fit distribution of crate date to match thin_section data. 
* exploratory_analysis.ipynb: Plots and compares the average absorbance sample 620 with one sample from crate data.
* hyperspectral_classifier.ipynb: Create and test the Classifier algorithm to predict the minerals on hyperspectral image.
* masking.ipynb: Preprocess the hyperspectral images and use the initial mask to create a NN model that differentiate cluster of rocks from background. Saves a binary file of a trained finer masking Neural Network model. 
* masking-Copy1.ipynb: Copy of masking.ipynb
* masking_compact.ipynb: Compact/Refined version of masking.ipynb
* multiple_input_model.ipynb: Apply the thin-section SVM classifier model (that was trained using automated-mineralogy) to the new crate data in two steps process: interpolatation and regression.
* codeNotes.md:
  
How to run:
* CNN_masker.ipynb:
* masking.ipynb:
* masking_compact.ipynb:
