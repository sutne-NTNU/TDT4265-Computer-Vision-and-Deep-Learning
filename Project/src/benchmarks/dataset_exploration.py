from utils.config import instantiate, LazyConfig
from utils import utils


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[
            :-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_dataloader(config, dataset_name):
    dataloader = get_dataloader(config, dataset_name)

    labels = {}
    aspectRatios = {}  # Width / Height

    for batch in utils.progress_bar(dataloader, f"Analyzing Dataset: {dataset_name}"):
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

                # Increase counter for number of label occurrences
                if i not in labels:
                    labels[i] = 1
                else:
                    labels[i] = labels[i] + 1

                # find aspect ratio of box and add it to dict list
                x_min, y_min, x_max, y_max = box
                width = (x_max - x_min)*image_width
                height = (y_max - y_min)*image_height
                aspectRatio = width/height

                if i not in aspectRatios:
                    aspectRatios[i] = [aspectRatio]
                else:
                    aspectRatios[i].append(aspectRatio)

    # Print Results for each label
    for i in sorted(labels.keys()):
        # Calculate average
        averageAspectRatio = sum(aspectRatios[i]) / len(aspectRatios[i])

        print(
            f"{i} {config.label_map[i]}: Occurrences: {labels[i]} Average Aspect Ratio: {averageAspectRatio}"
        )


if __name__ == '__main__':
    config = LazyConfig.load("model/configs/tdt4265.py")
    config.train.batch_size = 1

    analyze_dataloader(config, "train")
    analyze_dataloader(config, "val")
