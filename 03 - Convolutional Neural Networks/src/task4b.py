import matplotlib.pyplot as plt
from PIL import Image
import torchvision

from utils import torch_to_numpy_image


def resize_and_normalize(image: Image):
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return image_transform(image)[None]


if __name__ == "__main__":
    # Load image
    image = Image.open("images/zebra.jpg")
    image = resize_and_normalize(image)

    # Load the model from ResNet18
    model = torchvision.models.resnet18(pretrained=True)

    # Pass image through first layer once and get weights and activations
    activation = model.conv1(image)
    weight = model.conv1.weight.data.cpu()

    # Convert to images
    activation = [torch_to_numpy_image(a) for a in activation[0]]
    weight = [torch_to_numpy_image(w) for w in weight]

    indices = [14, 26, 32, 49, 52]
    # Plot weight and activation of the specified indices
    rows, cols, size = 2, len(indices), 5
    plt.figure(figsize=(size * cols, size * rows))
    for n, i in enumerate(indices, 1):
        # Plot Weight
        plt.subplot(rows, cols, n)
        plt.title(f"{i}", fontdict={"size": 25, "weight": "bold"})
        plt.imshow(weight[i])
        plt.axis("off")
        # Plot Activation
        plt.subplot(rows, cols, n + cols)
        plt.imshow(activation[i], cmap='gray')
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task4b.png")
