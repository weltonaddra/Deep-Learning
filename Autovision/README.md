# Autovision

## Intro
Welcome to Autovision, a Pnemonia vision learning Pytorch project.
Our model is trained on a large dataset to classify Xray images as either normal,
viral pneumonia, or bacterial pneumonia.

## Description
AutoVision is a pnuemonia vision learning project built with PyTorch and ResNet-50.
* Loads and preprocesses chest x-ray images into train/val/test splits
* Trains a ResNet-50 convolutional neual network, which is a popular deep learning model for image classification.
* Provides a GUI for loading a single image and predicting whether it is Normal, Viral Pneumonia, or Bacterial Pneumonia using trained model weights.

The project is organized into the following componenets:
* dataset.py: dataset loading, transforms, and dataloaders.
* trainer.py: model building, training, testing, and visualization.
* visuals.py: configuration for paths and hyperparameters.
* main.py: CLI entry point for training/testing
* AutoVisionGUI.py: Tkinter GUI for interactive inference. 

## Running the Model

To run the model you must:

### Install the required libraries:

---

- pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
- pip install numpy
- pip install matplotlib
- pip install pillow

---

The download the model weights from(file size was too big for github):
https://drive.google.com/file/d/11hzBnsYGyYy9UYdLWYfh899W3SwSym5c/view?usp=sharing

Clone all files from the github including chest xray files.

Run either the "main" file through the terminal, or the AutovisionGui file.
