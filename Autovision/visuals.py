import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from config import Config


class Visualizer:
    """Handles all image and grid visualization."""

    @staticmethod
    def imshow(inp, title=None, mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD):
        if isinstance(inp, torch.Tensor):
            inp = inp.detach().cpu()
        img = inp.numpy().transpose((1, 2, 0))
        img = np.clip((img * np.array(std)) + np.array(mean), 0, 1)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis('off')
        if title is not None:
            plt.title(title)
        plt.show()

    @staticmethod
    def imshow(img, ax=None):
        """Show image on the given matplotlib axis."""
        img = img.permute(1, 2, 0).numpy()  # Convert from tensor (C,H,W) to (H,W,C)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
        if ax is None:
            plt.imshow(img)
        else:
            ax.imshow(img)

    @staticmethod
    def show_batch(dataloader, class_names):
        inputs, classes = next(iter(dataloader))
        grid = torchvision.utils.make_grid(inputs, nrow=4, padding=2)
        Visualizer.imshow(grid, title=", ".join([class_names[c] for c in classes]))
