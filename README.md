# AutoVision

A pneumonia detection system using deep learning to classify chest X-ray images as normal, viral pneumonia, or bacterial pneumonia.

---

## Overview

AutoVision is a computer vision project built with PyTorch and ResNet-50 that provides intelligent pneumonia detection from chest X-ray images. The system includes both command-line and graphical user interfaces for model inference.

### Key Features
- **Image Classification**: Classifies X-ray images into three categories (Normal, Viral Pneumonia, Bacterial Pneumonia)
- **Deep Learning Model**: Utilizes ResNet-50 convolutional neural network
- **Dual Interface**: Both CLI and GUI options for model interaction
- **Pre-trained Weights**: Includes trained model for immediate use

---

## Project Structure

```
AutoVision/
├── dataset.py           # Dataset loading, transforms, and dataloaders
├── trainer.py           # Model building, training, testing, and visualization
├── visuals.py           # Configuration for paths and hyperparameters
├── main.py              # CLI entry point for training/testing
├── AutoVisionGUI.py     # Tkinter GUI for interactive inference
├── model_weights/       # Trained model weights (download separately)
└── chest_xray/          # Dataset directory
```

---

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Git for cloning the repository

### Install Required Libraries

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy
pip install matplotlib
pip install pillow
```

### Download Model Weights

Due to file size limitations, download the trained model weights separately:

**Download Link**: [Google Drive](https://drive.google.com/file/d/11hzBnsYGyYy9UYdLWYfh899W3SwSym5c/view?usp=sharing)

1. Download the model weights file
2. Place it in the `model_weights/` directory
3. Ensure the file is properly named for the application to locate

### Clone Repository

```bash
git clone <repository-url>
cd AutoVision
```

---

## Usage

### Command Line Interface

Run the model through the terminal:

```bash
python main.py
```

### Graphical User Interface

Launch the interactive GUI:

```bash
python AutoVisionGUI.py
```

### Model Operation

1. **Load Image**: Use either interface to load a chest X-ray image
2. **Predict**: The model will analyze and classify the image
3. **Results**: View classification as Normal, Viral Pneumonia, or Bacterial Pneumonia

---

## Technical Details

### Model Architecture
- **Base Model**: ResNet-50 (pre-trained on ImageNet)
- **Input**: Chest X-ray images (preprocessed to 224x224)
- **Output**: 3-class classification (Normal, Viral, Bacterial)
- **Framework**: PyTorch

### Dataset
- **Source**: Chest X-ray collection
- **Splits**: Training, validation, and testing sets
- **Preprocessing**: Standard image transforms and normalization

---

## Project Components

| File | Description |
|------|-------------|
| `dataset.py` | Handles dataset loading, data transforms, and creates PyTorch dataloaders |
| `trainer.py` | Contains model building logic, training procedures, testing, and result visualization |
| `visuals.py` | Configuration file for file paths, hyperparameters, and model settings |
| `main.py` | Command-line interface for model training, testing, and inference |
| `AutoVisionGUI.py` | Tkinter-based graphical user interface for interactive image classification |

---

## Authors

**Team AutoVision**
- Welton Addra
- Russell Bledsoe
- Evan Bradshaw
- Wyatt Bridges
- Moana Leo
- Brent Lewis

---

## License

This project is developed for educational and research purposes in medical image analysis.

---

*AutoVision - Intelligent Pneumonia Detection*
