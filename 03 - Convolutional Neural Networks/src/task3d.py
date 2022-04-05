from torch import nn
import numpy as np
from dataloaders import load_cifar10
from trainer import Trainer
from plotter import plot_loss_comparison
from utils import set_seeds
from task2 import Task2Model
from task3aModel2 import Task3aModel2


class Task3Model2NoDropout(Task2Model):
    """
    This model extends the model from task 2
    so if no changes are made to a variable/function, they are kept the same as in task 2
    """

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            # Layer 4
            nn.Linear(self.num_output_features, 256),
            nn.ReLU(),
            # Layer 5
            nn.Linear(256, 128),
            nn.ReLU(),
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

    # Duplicate of Model 2 from task 3a
    set_seeds(0)
    print(f"""
    Task 3d - With Dropout
    """)
    with_dropout = Trainer(
        Task3aModel2(),
        batch_size=64,
        learning_rate=0.15,
        early_stop_count=np.inf,  # don't early stop
        dataloaders=load_cifar10(batch_size=64),
    )
    with_dropout.train()
    res_with = with_dropout.test()

    # Exactly the same model but without dropout to see its effect
    set_seeds(0)
    print(f"""
    Task 3d - Without Dropout
    """)
    without_dropout = Trainer(
        Task3Model2NoDropout(),
        batch_size=64,
        learning_rate=0.15,
        early_stop_count=np.inf,  # don't early stop
        dataloaders=load_cifar10(batch_size=64),
    )
    without_dropout.train()
    res_without = without_dropout.test()

    # Plot comparison between the two
    plot_loss_comparison(
        trainer_with=with_dropout,
        trainer_without=without_dropout,
        accuracies_with=res_with,
        accuracies_without=res_without,
        diff="Dropout",
        filename="task3d.png"
    )
