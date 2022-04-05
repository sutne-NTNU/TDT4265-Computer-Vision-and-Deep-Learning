import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes
import typing


def area(xmin, ymin, xmax, ymax) -> float:
    """ Calculate non-negative area between points"""
    if xmin > xmax or ymin > ymax:
        return 0  # area would be negative, for intersecion this means no overlap
    return (xmax - xmin) * (ymax - ymin)


def calculate_iou(prediction_box, gt_box) -> float:
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    xmin_pr, ymin_pr, xmax_pr, ymax_pr = prediction_box
    xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_box

    intersection = area(
        xmin=max(xmin_pr, xmin_gt),
        ymin=max(ymin_pr, ymin_gt),
        xmax=min(xmax_pr, xmax_gt),
        ymax=min(ymax_pr, ymax_gt)
    )
    union = area(*prediction_box) + area(*gt_box) - intersection

    return intersection / union


def calculate_precision(num_tp: int, num_fp: int, num_fn: int) -> float:
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp: int, num_fp: int, num_fn: int) -> float:
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    return num_tp / (num_tp + num_fn)


def get_all_box_matches(prediction_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]

    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # a box_match = (prediction_box, gt_box)
    box_matches: typing.List[tuple[tuple, tuple]] = []
    # list of gt_boxes that hasn't been matched yet
    gt_boxes_left = list(gt_boxes)

    # for each prediction box
    for prediction_box in prediction_boxes:
        # find the ground truth box that is the best match
        best_match, best_iou = -1, -1
        for i, gt_box in enumerate(gt_boxes_left):
            iou = calculate_iou(prediction_box, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_match, best_iou = i, iou

        # add best match to list if there was a match
        if best_match != -1:
            gt_box = gt_boxes_left.pop(best_match)
            box_matches.append((prediction_box, gt_box))

    # Create numpy arrays of matched ground truth and prediction boxes
    prediction_boxes = np.array(
        [match[0] for match in box_matches], dtype=object
    )
    gt_boxes = np.array(
        [match[1] for match in box_matches], dtype=object
    )
    return prediction_boxes, gt_boxes


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, "false_neg": int}
    """
    matches = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    prediction_matches, gt_matches = matches

    true_positive = len(gt_matches)
    false_positive = len(prediction_boxes) - true_positive
    false_negative = len(gt_boxes) - true_positive

    return {"true_pos": true_positive, "false_pos": false_positive, "false_neg": false_negative}


def calculate_precision_recall_all_images(all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    true_positives, false_positives, false_negatives = 0, 0, 0

    for prediction_boxes, gt_boxes in zip(all_prediction_boxes, all_gt_boxes):
        image_result = calculate_individual_image_result(
            prediction_boxes, gt_boxes, iou_threshold
        )
        true_positives += image_result['true_pos']
        false_positives += image_result['false_pos']
        false_negatives += image_result['false_neg']

    return (
        calculate_precision(true_positives, false_positives, false_negatives),
        calculate_recall(true_positives, false_positives, false_negatives)
    )


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    confidence_thresholds = np.linspace(0, 1, 500)  # Approximation

    precisions, recalls = [], []
    for confidence_threshold in confidence_thresholds:
        confident_prediction_boxes = []

        for box_scores, predicted_boxes in zip(confidence_scores, all_prediction_boxes):
            index = box_scores >= confidence_threshold
            prediction_box = predicted_boxes[index]
            confident_prediction_boxes.append(prediction_box)

        precision, recall = calculate_precision_recall_all_images(
            confident_prediction_boxes, all_gt_boxes, iou_threshold
        )
        precisions.append(precision)
        recalls.append(recall)

    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls, mean_average_precision):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    plt.title(
        f"Precision Recall Curve - Mean average precision: {mean_average_precision:.4f}"
    )
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    recall_levels = np.linspace(0, 1.0, 11)

    average_precisions = []
    for recall_level in recall_levels:
        if any(recalls >= recall_level):
            average_precisions.append(max(precisions[recalls >= recall_level]))
        else:
            average_precisions.append(0)
    return sum(average_precisions) / len(average_precisions)


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    mean_average_precision = calculate_mean_average_precision(
        precisions, recalls)
    plot_precision_recall_curve(precisions, recalls, mean_average_precision)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
