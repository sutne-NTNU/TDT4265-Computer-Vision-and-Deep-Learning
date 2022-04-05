import torch
from dataloaders import load_cifar10

from trainer import Trainer
from plotter import plot_loss_and_accuracy
from utils import set_seeds
from task2 import Task2Model


if __name__ == "__main__":
    set_seeds(0)

    print(f"""
    Task 3a - Model 1
        - (x) ADAM optimizer 
        - (x) Learning Rate 0.00049
    """)

    trainer = Trainer(
        Task2Model(),
        batch_size=64,
        learning_rate=0.00049,
        optimizer=torch.optim.Adam,
        dataloaders=load_cifar10(batch_size=64),
    )
    trainer.train()

    plot_loss_and_accuracy(trainer, "task3a-model1.png")
