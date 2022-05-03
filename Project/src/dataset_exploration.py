import matplotlib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from collections import OrderedDict
import pathlib

from tops.config import instantiate, LazyConfig
from ssd.utils import batch_collate, batch_collate_val
import tops
from configs.utils import get_output_dir


matplotlib.rcParams.update(
    {
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "k",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
    }
)


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = (
            cfg.data_train.dataset.transform.transforms[:-1]
        )
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_dataloader(config, dataset_name) -> OrderedDict:
    dataloader = get_dataloader(config, dataset_name)

    label_results = OrderedDict()
    for i in list(config.label_map.keys())[1:]:
        label_results[i] = {
            "occurrences": 0,
            "widths": [],
            "heights": [],
            "aspectRatios": [],
            "name": config.label_map[i],
        }

    for batch in tops.misc.progress_bar(
        dataloader, f"Analyzing Dataset: {dataset_name}"
    ):
        # Batch Contains:
        #   'image'     - image data
        #   'image_id'  - id of image (number)
        #   'width'     - width of image
        #   'height'    - height of image
        #   'labels'    - labels for the current image
        #   'boxes'     - boxes corresponsing to the labels
        image_width = float(batch["width"])
        image_height = float(batch["height"])

        for image_labels, image_boxes in zip(batch["labels"], batch["boxes"]):
            for label, box in zip(image_labels, image_boxes):
                i = int(label)

                # Calculating values
                x_min, y_min, x_max, y_max = box
                width = float((x_max - x_min) * image_width)
                height = float((y_max - y_min) * image_height)
                aspectRatio = width / height

                label_results[i]["occurrences"] += 1
                label_results[i]["widths"].append(width)
                label_results[i]["heights"].append(height)
                label_results[i]["aspectRatios"].append(aspectRatio)

    return label_results


def verify_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def plot_results(data: dict, output_dir: str):

    figsize = (10, 6)
    colors = [
        "#1f77b4",  # car
        "#ff7f0e",  # truck
        "#2ca02c",  # bus
        "#7f7f7f",  # motorcycle
        "#9467bd",  # bicycle
        "#8c564b",  # scooter
        "#d62728",  # person
        "#e377c2",  # rider
    ]

    # Occurrences
    verify_dir(output_dir)
    plt.figure(figsize=figsize)
    plt.title("Number of Occurrences of Each Label")
    for i in data.keys():
        x = data[i]["name"]
        y = data[i]["occurrences"]
        bar = plt.bar(x, y, color=colors[i - 1])
        plt.bar_label(bar)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "occurrences.png"))

    # Size
    size_output_dir = os.path.join(output_dir, "sizes/")
    verify_dir(size_output_dir)
    for i in data.keys():
        plt.figure(figsize=figsize)
        name = data[i]["name"]
        plt.title(f"Width and Height for Bounding Boxes (in pixels) for {name}")
        widths = data[i]["widths"]
        heights = data[i]["heights"]
        plt.xlim((0, 400))
        plt.ylim((0, 130))
        plt.scatter(widths, heights, label=name, color=colors[i - 1])
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.savefig(os.path.join(size_output_dir, f"{name}.png"))

    # Aspect Ratios
    plt.figure(figsize=figsize)
    plt.title(f"Stacked Aspect Ratios for Bounding Boxes for all labels")
    values = [data[i]["aspectRatios"] for i in data.keys()]
    labels = [data[i]["name"] for i in data.keys()]
    plt.hist(values, density=True, stacked=True, bins=50, label=labels, color=colors)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aspect-ratios.png"))


if __name__ == "__main__":
    root = pathlib.Path("../plots/dataset_exploration/")

    config = LazyConfig.load("configs/baseline.py")
    config.train.batch_size = 1
    results = analyze_dataloader(config, "train")
    plot_results(results, os.path.join(root, "train_tdt4265/"))
    results = analyze_dataloader(config, "val")
    plot_results(results, os.path.join(root, "val_tdt4265/"))

    config = LazyConfig.load("configs/updated_dataset.py")
    config.train.batch_size = 1
    results = analyze_dataloader(config, "train")
    plot_results(results, os.path.join(root, "train_tdt4265_extended/"))
