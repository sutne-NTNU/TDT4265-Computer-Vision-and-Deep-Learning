import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import pathlib

from trainer import Trainer

plot_path = pathlib.Path("results")
plot_path.mkdir(exist_ok=True)


def plot(hist_dict: dict, label: str = None, num_average=1, plot_variance=True):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
        num_average: Number of points to average plot
    """
    global_steps = list(hist_dict.keys())
    loss = list(hist_dict.values())
    if num_average == 1 or not plot_variance:
        plt.plot(global_steps, loss, label=label)
        return

    num_points = len(loss) // num_average
    mean_loss = []
    loss_std = []
    steps = []
    for i in range(num_points):
        points = loss[i*num_average:(i+1)*num_average]
        step = global_steps[i*num_average + num_average//2]
        mean_loss.append(np.mean(points))
        loss_std.append(np.std(points))
        steps.append(step)
    plt.plot(steps, mean_loss,
             label=f"{label} (mean over {num_average} steps)")
    plt.fill_between(
        steps, np.array(mean_loss) -
        np.array(loss_std), np.array(mean_loss) + loss_std,
        alpha=.2, label=f"{label} (variance over {num_average} steps)")


def plot_loss_and_accuracy(trainer: Trainer, filename: str):
    """
    Plots the training and validation loss in one graph, and the validation accuracy in another.

    Also adds the final accuracies as text to the plot.
    """
    train_loss, train_accuracy = trainer.training_history()
    validation_loss, validation_accuracy = trainer.validation_history()

    plt.figure(figsize=(16, 10))
    # Loss
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    plot(train_loss, label="Training loss", num_average=50)
    plot(validation_loss, label="Validation loss")
    plt.legend()
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plot(validation_accuracy, label="Validation Accuracy")
    plt.legend()

    plt.tight_layout()

    train, val, test = trainer.test()
    # Add final results text box
    add_text_box(f"""Final Accuracy Results:
    Training:   {100*train[1]:.2f}%
    Validation: {100*val[1]:.2f}%
    Test:       {100*test[1]:.2f}%""")

    plt.savefig(plot_path.joinpath(filename))


def plot_loss_comparison(
    trainer_with: Trainer,
    trainer_without: Trainer,
    diff: str,
    accuracies_with: tuple,
    accuracies_without: tuple,
    filename: str,
):
    """
    """
    train_loss_with, _ = trainer_with.training_history()
    train_loss_without, _ = trainer_without.training_history()
    validation_loss_with, _ = trainer_with.validation_history()
    validation_loss_without, _ = trainer_without.validation_history()

    plt.figure(figsize=(16, 10))
    # Train Loss
    plt.subplot(1, 2, 1)
    plt.title("Training Loss")
    plot(train_loss_with, label=f"With {diff}", num_average=50)
    plot(train_loss_without, label=f"Without {diff}", num_average=50)
    plt.legend()
    # Add final accuracies for model without
    train, val, test = accuracies_without
    add_text_box(f"""Accuracy Results Without {diff}:
    Training:   {100*train[1]:.2f}%
    Validation: {100*val[1]:.2f}%
    Test:       {100*test[1]:.2f}%""", loc=3)

    # Validation Loss
    plt.subplot(1, 2, 2)
    plt.title("Validation Loss")
    plot(validation_loss_with, label=f"With {diff}")
    plot(validation_loss_without, label=f"Without {diff}")
    plt.legend()
    # Add final accuracies for model with
    train, val, test = accuracies_with
    add_text_box(f"""Accuracy Results With {diff}:
    Training:   {100*train[1]:.2f}%
    Validation: {100*val[1]:.2f}%
    Test:       {100*test[1]:.2f}%""")

    plt.tight_layout()

    plt.savefig(plot_path.joinpath(filename))


def add_text_box(text: str, loc=4):
    """ Place textbox in plot, default location=4 (bottom right) """
    text_box = AnchoredText(
        text,
        frameon=True,
        loc=loc,
        pad=0.5,
        prop=dict(fontname="monospace", size=14)
    )
    plt.setp(
        text_box.patch,
        facecolor='white',
        alpha=0.3
    )
    plt.gca().add_artist(text_box)
