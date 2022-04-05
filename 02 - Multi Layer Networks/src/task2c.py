import matplotlib.pyplot as plt
import numpy as np

import utils
from task2ab import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
from trainer import BaseTrainer
np.random.seed(0)

results_file = 'results/task2_final_results.txt'


def write_results_to_file(description: str, model: SoftmaxModel):
    t_loss = cross_entropy_loss(Y_train, model.forward(X_train))
    v_loss = cross_entropy_loss(Y_val, model.forward(X_val))
    t_acc = calculate_accuracy(X_train, Y_train, model)
    v_acc = calculate_accuracy(X_val, Y_val, model)
    utils.write_results_to_file(
        results_file, description, t_loss, v_loss, t_acc, v_acc
    )


def get_predictions(output):
    ''' Takes the output of a model and returns a one_hot_encoded result of same shape '''
    index_of_predictions = np.array([np.argmax(output, axis=1)]).T
    return one_hot_encode(index_of_predictions, output.shape[1])


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    output = model.forward(X)
    predictions = get_predictions(output)

    num_predictions = output.shape[0]
    num_correct_predictions = np.count_nonzero(targets * predictions)

    return num_correct_predictions/num_predictions


class SoftmaxTrainer(BaseTrainer):

    def __init__(
            self,
            momentum_gamma: float = 0.0,
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.use_momentum = momentum_gamma != 0.0
        self.momentum_gamma = momentum_gamma
        # Previous gradients
        self.prev_grads = [np.zeros_like(w) for w in self.model.weights]

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # Forward
        output = self.model.forward(X_batch)
        # Backward
        self.model.backward(X_batch, output, Y_batch)
        # Gradient Descent
        for i, grad in enumerate(self.model.grads):
            if self.use_momentum:
                grad = grad + self.momentum_gamma * self.prev_grads[i]
            self.model.weights[i] -= self.learning_rate * grad
            self.prev_grads[i] = grad
        # Return loss
        return cross_entropy_loss(Y_batch, output)

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    utils.setup(results_file)

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train, X_val = pre_process_images(X_train), pre_process_images(X_val)
    Y_train, Y_val = one_hot_encode(Y_train, 10), one_hot_encode(Y_val, 10)

    # Create and Train the model
    model = SoftmaxModel(
        neurons_per_layer=[64, 10],
    )
    trainer = SoftmaxTrainer(
        model=model,
        X_train=X_train,
        X_val=X_val,
        Y_train=Y_train,
        Y_val=Y_val,
        batch_size=32,
        shuffle_dataset=True,
        learning_rate=.1,
    )
    train_history, val_history = trainer.train(num_epochs=50)

    utils.write_markdown_header(results_file)
    write_results_to_file('Hidden Layer With 64 Neurons', model)

    # Plot History
    plt.figure(figsize=(16, 10))
    # Loss
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    plt.title('Training and Validation Loss')
    utils.plot_loss(train_history["loss"], "Training Loss", 50)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend(loc='upper right')
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.875, .99])
    plt.title('Training and Validation Accuracy')
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("results/task2c.png")
