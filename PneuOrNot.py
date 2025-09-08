#Set up and downloads 
# pip install -Uqq fastai  (in the terminal)

#Data loaders are used to load data into the model to train it.
from fastai.vision.all import *
path = Path('chest_xray')
train_set = Path('chest_xray/train')
test_set = Path('chest_xray/train')
val_set = Path('chest_xray/train')

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)