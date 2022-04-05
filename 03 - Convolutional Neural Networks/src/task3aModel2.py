from torch import nn
from dataloaders import load_cifar10

from trainer import Trainer
from plotter import plot_loss_and_accuracy
from utils import set_seeds
from task2 import Task2Model


class Task3aModel2(Task2Model):
    """
    This model extends the model from task 2
    so if no changes are made to a variable/function, they are kept the same as in task 2
    """

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            # Layer 4
            nn.Linear(self.num_output_features, 256),
            nn.ReLU(),
            # Layer 5
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            # Layer 6
            nn.Linear(128, 64),
            nn.ReLU(),
            # Layer 7
            nn.Linear(64, 32),
            nn.ReLU(),
            # Layer 8
            nn.Linear(32, self.num_classes),
        )


if __name__ == "__main__":
    set_seeds(0)

    print(f"""
    Task 3a - Model 2
        - (x) More Classification Layers
        - (x) Learning Rate 0.15
        - (x) Dropout
    """)

    trainer = Trainer(
        model=Task3aModel2(),
        batch_size=64,
        learning_rate=0.15,
        dataloaders=load_cifar10(batch_size=64),
    )
    trainer.train()

    plot_loss_and_accuracy(trainer, "task3a-model2.png")
