from torch import nn
from dataloaders import load_cifar10

from trainer import Trainer
from plotter import plot_loss_and_accuracy
from utils import set_seeds


class Task2Model(nn.Module):

    def __init__(self, image_channels=3, num_classes=10):
        """
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.image_channels = image_channels
        self.num_classes = num_classes

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(image_channels, 32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16 x 16
            # Layer 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8 x 8
            # Layer 3
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4 x 4
        )

        # After convolution shapes are width=4, height=4 with 128 outputs
        self.num_output_features = 4 * 4 * 128

        # Define Fully Connected Layers (takes result of convolution and classifies them)
        self.classifier = nn.Sequential(
            # Layer 4
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            # Layer 5
            nn.Linear(64, self.num_classes),
        )

    def forward(self, X):
        """
        Performs a forward pass through the model
            Args:
                x: Input images, shape: [batch_size, 3, 32, 32]
        """
        out = self.feature_extractor(X)
        out = out.view(-1, self.num_output_features)
        return self.classifier(out)


if __name__ == "__main__":
    set_seeds(0)

    print("""
    Task 2
    """)

    trainer = Trainer(
        Task2Model(),
        batch_size=64,
        learning_rate=0.05,
        dataloaders=load_cifar10(batch_size=64),
    )
    trainer.train()

    plot_loss_and_accuracy(trainer, "task2.png")
