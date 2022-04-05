import matplotlib.pyplot as plt
import numpy as np
import os

import utils
from task2a import pre_process_images
from task3a import SoftmaxModel, one_hot_encode
from task3 import SoftmaxTrainer
np.random.seed(0)


class L2SoftmaxModel(SoftmaxModel):
    """
    Extends the SoftMaxModel's backward pass to include the L2 regularization.
    """

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        super().backward(X, outputs, targets)

        # Task 4b
        self.grad += 2 * self.l2_reg_lambda * self.w


def train_with_lambda(l2_lambda: float):
    """Trains a model with L2 regularization with the specified Lambda value

    Args:
        l2_lambda (float): The lambda value

    Returns:
        model, training_history, validation_history
    """
    model = L2SoftmaxModel(l2_reg_lambda=l2_lambda)
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    training_history, validation_history = trainer.train(num_epochs)
    return model, training_history, validation_history


if __name__ == "__main__":
    # hyperparameters
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    if not os.path.exists('results'):
        os.makedirs('results')

    # Task 4b
    # Train models with L2 regularization with different lambdas and
    # visualize the weights
    l2_lambdas = [0, 2]
    rows, cols, size, cmap = 2, 10, 4, "gray"
    plt.figure(figsize=(cols*size, rows*size))
    for row, l2_lambda in enumerate(l2_lambdas):
        model, training, validation = train_with_lambda(l2_lambda)
        for i in range(cols):
            image_index = (row * cols) + (i + 1)
            plt.subplot(rows, cols, image_index)
            plt.imshow(model.w[:, i][1:].reshape(28, 28), cmap=cmap)
            plt.axis("off")
    plt.tight_layout()
    plt.savefig("results/task4b_model_weights.png")

    # Task 4c
    # Plotting of validation accuracy for different values of lambda
    l2_lambdas = [2.0, 0.2, 0.02, 0.002]
    plt.figure()
    for l2_lambda in l2_lambdas:
        model, training, validation = train_with_lambda(l2_lambda)
        utils.plot_loss(
            validation["accuracy"],
            '$\\bf{\lambda = %s}$' % l2_lambda
        )
    plt.ylim([0.625, .925])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("results/task4c_l2_validation_accuracy.png")

    # Task 4e
    # Plotting length of weight vector for different lambda values
    plt.figure()
    l2_norms = []
    for l2_lambda in l2_lambdas:
        model, training, validation = train_with_lambda(l2_lambda)
        l2_norm = np.sum(np.square(model.w))
        l2_norms.append(l2_norm)
    plt.plot(l2_lambdas, l2_norms, 'r:')
    plt.scatter(l2_lambdas, l2_norms)
    plt.xlabel("$\lambda$")
    plt.ylabel("$L_2\ norm,\ ||w||^2$")
    plt.savefig("results/task4e_l2_vector_lengths.png")
