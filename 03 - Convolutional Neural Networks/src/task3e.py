from torchvision import transforms
from torch import nn
import torch

from dataloaders import load_cifar10, mean, std
from utils import set_seeds
from plotter import plot_loss_and_accuracy
from trainer import Trainer
from task2 import Task2Model


class Task3eModel(Task2Model):

    def __init__(self):
        super().__init__()
        in_channels = self.image_channels
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Layer 2
            nn.Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 x 16
            # Layer 3
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Layer 4
            nn.Conv2d(64, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 x 8
            # Layer 5
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Layer 6
            nn.Conv2d(128, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 x 4
        )

        self.num_output_features = 128 * 4 * 4

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),

            # Layer 7
            nn.Linear(self.num_output_features, 128),
            nn.ReLU(),
            # Layer 8
            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Dropout(0.1),
            # Layer 9
            nn.Linear(64, 64),
            nn.ReLU(),
            # Layer 10
            nn.Linear(64, self.num_classes),
        )


if __name__ == "__main__":
    set_seeds(0)

    print(f"""
    Task 3e
        - (x) More Feature Extraction Layers
        - (x) Batch Normalization
        - (x) More Classification Layers
        - (x) Dropout
        - (x) ADAM optimizer
        - (x) Learning Rate 0.0006
        - (x) Data Augmentation
    """)

    trainer = Trainer(
        Task3eModel(),
        batch_size=32,
        learning_rate=0.0006,
        optimizer=torch.optim.Adam,
        dataloaders=load_cifar10(
            batch_size=32,
            transform_train_val=transforms.Compose([
                # Apply some random transformations
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                # Output and normalize to tensor
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
        )
    )
    trainer.train()

    plot_loss_and_accuracy(trainer, "task3e.png")
