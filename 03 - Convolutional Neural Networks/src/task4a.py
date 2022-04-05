from torchvision import transforms
from torch import nn
import torchvision
import torch

from dataloaders import load_cifar10
from trainer import Trainer
from plotter import plot_loss_and_accuracy


class TransferModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10)
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    # Resize and normalize with correct values
    resnet_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    # Train the model (should converge before 5 epochs)
    trainer = Trainer(
        TransferModel(),
        epochs=5,
        batch_size=32,
        learning_rate=0.0005,
        optimizer=torch.optim.Adam,
        dataloaders=load_cifar10(
            batch_size=32,
            transform_train_val=resnet_transform,
            transform_test=resnet_transform,
        ),
    )
    trainer.train()

    plot_loss_and_accuracy(trainer, "Task4a.png")
