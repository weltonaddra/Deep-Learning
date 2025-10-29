import os
import time
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from tempfile import TemporaryDirectory
from torch.backends import cudnn
from config import Config
from visuals import Visualizer


class ModelTrainer:
    # Used to build the model, train it, test it and visualize results.
    # Expects a ChestXRayDataset object.
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = dataset.device
        self.model = self._build_model()
        # optimizer/criterion/scheduler/scaler will be set here
        self._setup_optimizers_and_criterion()
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def _setup_optimizers_and_criterion(self):
        # Compute class weights, create criterion, optimizer and OneCycleLR scheduler.
        weight_tensor = None
        try:
            train_ds = self.dataset.datasets['train']
            base_ds = train_ds.dataset if isinstance(train_ds, torch.utils.data.Subset) else train_ds
            targets = getattr(base_ds, 'targets', None)

            if targets is None:
                targets = [s[1] for s in getattr(base_ds, 'samples', [])]

            counts = np.bincount(targets, minlength=len(self.dataset.class_names))
            counts = np.where(counts == 0, 1, counts)  # avoid division by zero

            class_weights = (1.0 / counts).astype(np.float32)
            class_weights = class_weights * (len(class_weights) / class_weights.sum())  # normalize
            weight_tensor = torch.tensor(class_weights, device=self.device)
        except Exception:
            weight_tensor = None

        self.criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor) if weight_tensor is not None else torch.nn.CrossEntropyLoss()

        # optimizer - AdamW good default for fine-tuning
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=Config.LR, weight_decay=1e-4)

        # OneCycleLR needs steps_per_epoch
        train_loader = self.dataset.dataloaders.get('train')
        steps_per_epoch = max(1, len(train_loader)) if train_loader is not None else 1
        try:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=Config.LR,
                steps_per_epoch=steps_per_epoch,
                epochs=Config.NUM_EPOCHS,
                pct_start=0.1,
                anneal_strategy='cos',
            )
            self.step_scheduler_per_batch = True
        except Exception:
            # fallback to StepLR per epoch
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=Config.STEP_SIZE, gamma=Config.GAMMA)
            self.step_scheduler_per_batch = False

    def _build_model(self):
        # Build ResNet18 and adapt final fc to num classes
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            model = torchvision.models.resnet18(weights=weights)
        except Exception:
            # older/newer torchvision compatibility fallback
            model = torchvision.models.resnet18(pretrained=True)

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(self.dataset.class_names))
        return model.to(self.device)

    def load_checkpoint(self, path, map_location=None, reinit_fc_if_mismatch=True):
        # Safe checkpoint loader: loads matching keys and optionally reinitializes fc
        map_location = map_location or self.device
        ckpt = torch.load(path, map_location=map_location)
        # accept either full checkpoint or raw state_dict
        saved_sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        cur_sd = self.model.state_dict()

        matched = {k: v for k, v in saved_sd.items() if k in cur_sd and v.size() == cur_sd[k].size()}
        cur_sd.update(matched)
        self.model.load_state_dict(cur_sd)

        # reinit final fc if mismatch between saved and current
        if reinit_fc_if_mismatch and 'fc.weight' in saved_sd:
            saved_shape = saved_sd['fc.weight'].shape
            cur_shape = self.model.fc.weight.shape
            if saved_shape != cur_shape:
                in_features = self.model.fc.in_features
                out_features = len(self.dataset.class_names)
                self.model.fc = torch.nn.Linear(in_features, out_features).to(self.device)
                print(f"Reinitialized final fc: {in_features} -> {out_features} (checkpoint head {saved_shape} != model head {cur_shape})")

        print(f"Loaded {len(matched)} / {len(saved_sd)} tensors from {path}")

    def freeze_backbone(self):  
        """Freeze all layers except the final fc."""
        for name, param in self.model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def train_model(self, num_epochs=Config.NUM_EPOCHS, freeze_head_epochs=1, use_amp=True):
        """Train with optional freeze-then-unfreeze strategy, OneCycleLR (per-batch) and AMP."""
        since = time.time()
        cudnn.benchmark = True

        # prepare best model saving
        with TemporaryDirectory() as tempdir:
            best_model_path = os.path.join(tempdir, 'best_model_params.pt')
            torch.save(self.model.state_dict(), best_model_path)
            best_acc = 0.0

            # optionally freeze backbone for initial epochs
            if freeze_head_epochs > 0:
                self.freeze_backbone()
                print(f"Backbone frozen for {freeze_head_epochs} epoch(s). Training head only.")

            for epoch in range(num_epochs):
                print("")
                print(f"Epoch {epoch}/{num_epochs - 1}")
                print('-' * 10)

                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()
                    else:
                        self.model.eval()

                    running_loss = 0.0
                    running_corrects = 0

                    dataloader = self.dataset.dataloaders[phase]
                    for inputs, labels in dataloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            if use_amp and self.scaler is not None:
                                with torch.cuda.amp.autocast():
                                    outputs = self.model(inputs)
                                    _, preds = torch.max(outputs, 1)
                                    loss = self.criterion(outputs, labels)
                            else:
                                outputs = self.model(inputs)
                                _, preds = torch.max(outputs, 1)
                                loss = self.criterion(outputs, labels)

                            if phase == 'train':
                                if use_amp and self.scaler is not None:
                                    self.scaler.scale(loss).backward()
                                    self.scaler.step(self.optimizer)
                                    self.scaler.update()
                                else:
                                    loss.backward()
                                    self.optimizer.step()

                                # step per batch if OneCycleLR
                                if getattr(self, 'step_scheduler_per_batch', False):
                                    try:
                                        self.scheduler.step()
                                    except Exception:
                                        pass

                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data).item()

                    # scheduler stepping per epoch fallback
                    if not getattr(self, 'step_scheduler_per_batch', False):
                        try:
                            self.scheduler.step()
                        except Exception:
                            pass

                    # avoid division by zero
                    denom = max(1, self.dataset.dataset_sizes.get(phase, 0))
                    epoch_loss = running_loss / denom
                    epoch_acc = running_corrects / denom if denom > 0 else 0.0
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # save best model based on val acc
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model.state_dict(), best_model_path)

                # after freeze epochs, unfreeze
                if epoch + 1 == freeze_head_epochs:
                    self.unfreeze_all()
                    print("Unfroze all layers for fine-tuning.")

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:.4f}')

            # load best weights
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def visualize_model(self, num_images=3):
        was_training = self.model.training
        self.model.eval()
        images_shown = 0
        fig = plt.figure(figsize=(num_images * 4, 4))

        with torch.no_grad():
            for inputs, labels in self.dataset.dataloaders['val']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size(0)):
                    if images_shown >= num_images:
                        plt.tight_layout()
                        plt.show()
                        self.model.train(mode=was_training)
                        return
                    ax = plt.subplot(1, num_images, images_shown + 1)
                    ax.axis('off')
                    pred_label = self.dataset.class_names[preds[j]]
                    actual_label = self.dataset.class_names[labels[j]]
                    ax.set_title(f'Pred: {pred_label}\nActual: {actual_label}')
                    Visualizer.imshow(inputs.cpu().data[j], ax=ax)
                    images_shown += 1

        plt.tight_layout()
        plt.show()
        self.model.train(mode=was_training)

    def test_model(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.dataset.dataloaders['test']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy
