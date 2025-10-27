import os

class Config:
    """Global configuration constants."""
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    PAD_FILL_RGB  = tuple(int(m * 255) for m in IMAGENET_MEAN)

    DATA_DIR   = os.path.join("Class Files" , "chest_xray")
    BATCH_SIZE = 6
    NUM_WORKERS = 0
    IMG_SIZE = 224

    # Training hyperparameters
    NUM_EPOCHS = 25
    LR = 0.001
    MOMENTUM = 0.9
    STEP_SIZE = 7
    GAMMA = 0.1

    ## State Dict save file name 
   

