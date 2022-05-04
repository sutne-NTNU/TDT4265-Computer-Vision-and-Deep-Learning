import matplotlib
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
import pathlib

from tops.config import instantiate, LazyConfig
from ssd.utils import batch_collate, batch_collate_val
import tops


def verify_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


plot_output_dir = pathlib.Path("../plots/dataset_exploration/")

matplotlib.rcParams.update(
    {
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "k",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "figure.figsize": (10, 6),
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
    for label in list(config.label_map.keys())[1:]:
        label_results[label] = {
            "name": config.label_map[label],
            "occurrences": 0,
            "widths": [],
            "heights": [],
            "aspect_ratios": [],
        }

    for batch in tops.misc.progress_bar(dataloader, f"'{dataset_name}'\t"):
        # Batch Contains:
        #   'image'     - image data
        #   'image_id'  - id of image (number)
        #   'width'     - width of image
        #   'height'    - height of image
        #   'labels'    - labels in the current image
        #   'boxes'     - boxes corresponsing to the labels
        image_width = float(batch["width"])
        image_height = float(batch["height"])

        for image_labels, image_boxes in zip(batch["labels"], batch["boxes"]):
            for label, box in zip(image_labels, image_boxes):
                label = int(label)

                x_min, y_min, x_max, y_max = box
                width = float(x_max - x_min) * image_width
                height = float(y_max - y_min) * image_height
                aspectRatio = width / height

                label_results[label]["occurrences"] += 1
                label_results[label]["widths"].append(width)
                label_results[label]["heights"].append(height)
                label_results[label]["aspect_ratios"].append(aspectRatio)

    return label_results


def plot_results(data: dict, output_dir: str = None):
    if output_dir is None:
        output_dir = plot_output_dir
    else:
        output_dir = os.path.join(plot_output_dir, output_dir)
    verify_dir(output_dir)

    # want to make sure the same color is used for the same label in all plots
    colors = [
        "tab:blue",  # car
        "tab:brown",  # truck
        "tab:cyan",  # bus
        "tab:gray",  # motorcycle
        "tab:pink",  # bicycle
        "tab:green",  # scooter
        "tab:orange",  # person
        "tab:red",  # rider
    ]
    labels = [data[i]["name"] for i in data.keys()]

    # Occurrences
    values = [data[i]["occurrences"] for i in data.keys()]
    plt.figure()
    plt.bar_label(plt.bar(labels, values, color=colors))
    plt.title("Number of Occurrences of Each Label")
    plt.ylabel("Number of Occurrences")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/occurrences.png")
    plt.close()

    # Aspect Ratios
    values = [data[i]["aspect_ratios"] for i in data.keys()]
    plt.figure()
    plt.hist(values, stacked=True, bins=60, range=(0, 3), label=labels, color=colors)
    plt.title(f"Stacked Aspect Ratios for Bounding Boxes for all labels")
    plt.xlabel("Aspect Ratio (width/height)")
    plt.ylabel("Count (Stacked)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aspect-ratios.png")
    plt.close()

    # Size
    for i in data.keys():
        label = data[i]["name"]
        widths = data[i]["widths"]
        heights = data[i]["heights"]
        plt.figure()
        plt.scatter(widths, heights, label=label, color=colors[i - 1], marker=".")
        plt.title(f"Width and Height for Bounding Boxes for '{label}'")
        plt.xlim((0, 400))
        plt.ylim((0, 130))
        plt.xlabel("Width (pixels)")
        plt.ylabel("Height (pixels)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sizes-{label}.png")
        plt.close()


if __name__ == "__main__":
    config = LazyConfig.load("configs/baseline.py")
    config.train.batch_size = 1
    results = analyze_dataloader(config, "train")
    plot_results(results, "train_tdt4265")
    results = analyze_dataloader(config, "val")
    plot_results(results, "val_tdt4265")

    config = LazyConfig.load("configs/updated_dataset.py")
    config.train.batch_size = 1
    results = analyze_dataloader(config, "train")
    plot_results(results, "train_tdt4265_extended")
