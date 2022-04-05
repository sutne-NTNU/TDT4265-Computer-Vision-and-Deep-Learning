import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import math

from utils import torch_to_numpy_image
from task4b import resize_and_normalize

if __name__ == "__main__":
    # Load image
    image = Image.open("images/zebra.jpg")
    image = resize_and_normalize(image)

    # Load the model from ResNet18
    model = torchvision.models.resnet18(pretrained=True)

    # "Remove" last two layers of the model
    model = torch.nn.Sequential(*(list(model.children())[:-2]))

    # Perform a single forward pass through this new model
    activation = model.forward(image)
    assert activation.shape == (1, 512, 7, 7), activation.shape

    # Convert to images
    activation = [torch_to_numpy_image(a) for a in activation[0]]

    indices = range(0, 10)
    # Plot activation of the specified indices
    rows, cols, size = 2, math.ceil(len(indices)/2), 5
    plt.figure(figsize=(size * cols, size * rows))
    for n, i in enumerate(indices, 1):
        plt.subplot(rows, cols, n)
        plt.title(f"{i}", fontdict={"size": 25, "weight": "bold"})
        plt.imshow(activation[i])
        plt.axis("off")
    plt.subplots_adjust(top=0.75)
    plt.tight_layout()
    plt.savefig("results/task4c.png")
