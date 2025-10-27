from dataset import ChestXRayDataset
from trainer import ModelTrainer
from visuals import Visualizer
from config import Config
import torch
import os

def main():
    dataset = ChestXRayDataset(subset_fraction=1)  
    print("Classes:", dataset.class_names)

    trainer = ModelTrainer(dataset) 

    savedPath = "AutovisionVer1.pt"
    if os.path.exists(savedPath):
        trainer.load_checkpoint(savedPath)  # safe load

    trained_model = trainer.train_model()
    trainer.test_model()

    # Saves the models optimizer, scheduler and weights to train again later 
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict() if hasattr(trainer, 'optimizer') else None,
        'scheduler_state_dict': trainer.scheduler.state_dict() if hasattr(trainer, 'scheduler') else None,
    }, savedPath)

    trainer.visualize_model()
    

if __name__ == "__main__":
    main()






    # Preview batch
    # Visualizer.show_batch(dataset.dataloaders['train'], dataset.class_names)
