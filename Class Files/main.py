from dataset import ChestXRayDataset
from trainer import ModelTrainer
from visuals import Visualizer
import torch

def main():
    # Load data
    dataset = ChestXRayDataset(subset_fraction=1.0)

    # Preview batch
    # Visualizer.show_batch(dataset.dataloaders['train'], dataset.class_names)

    # Train and visualize results
    trainer = ModelTrainer(dataset)
    trained_model = trainer.train_model()

    trained_model.load_state_dict(torch.load("resnet18_pneumonia.pt"))
    trainer.visualize_model()

    print("Saving model...")
    torch.save(trained_model.state_dict(), "resnet18_pneumonia.pt")
    trained_model = trainer.train_model()
    trainer.visualize_model()

if __name__ == "__main__":
    main()

