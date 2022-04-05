import torch
import typing
import collections
import pathlib
import utils
import tqdm
from dataloaders import load_cifar10


def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    num_batches = len(dataloader)
    total_loss, total_accuracy = 0, 0

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            batch_size = Y_batch.shape[0]
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)

            output = model(X_batch)
            predictions = torch.argmax(output, axis=1)

            total_accuracy += (predictions == Y_batch).sum().cpu() / batch_size
            total_loss += loss_criterion(output, Y_batch).cpu()

    average_loss = total_loss / num_batches
    average_accuracy = total_accuracy / num_batches
    return average_loss, average_accuracy


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: typing.List[torch.utils.data.DataLoader],
        learning_rate: float,
        batch_size: int,
        epochs: int = 10,
        early_stop_count: int = 4,
        optimizer: torch.optim = torch.optim.SGD,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        # Initialize the model
        self.model: torch.nn.Module = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent is used by default
        self.optimizer = optimizer(
            self.model.parameters(),
            self.learning_rate
        )

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders

        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0

        # Tracking variables
        self.train_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()
        )
        self.val_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()
        )
        self.checkpoint_dir = pathlib.Path("checkpoints")

    def training_history(self):
        """ Return the Training History as tuple: (loss, accuracy)"""
        return self.train_history["loss"], self.train_history["accuracy"]

    def validation_history(self):
        """ Return the Validation History as tuple: (loss, accuracy)"""
        return self.val_history["loss"], self.val_history["accuracy"]

    def test(self):
        """
        Calculates the models Train, Validation and Test Loss and Accuracy

        Returns:
            Three tuples one for train, val and test, each containing: [0] loss, [1] accuracy
        """
        self.load_best_model()
        self.model.eval()
        train_loss, train_accuracy = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        val_loss, val_accuracy = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        test_loss, test_accuracy = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.model.train()
        return (train_loss, train_accuracy), (val_loss, val_accuracy), (test_loss, test_accuracy)

    def validation_step(self) -> tuple:
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.val_history["loss"][self.global_step] = validation_loss
        self.val_history["accuracy"][self.global_step] = validation_acc
        self.model.train()
        return validation_loss, validation_acc

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        val_loss = self.val_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(val_loss.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            return True
        return False

    def train_step(self, X_batch, Y_batch):
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
        X_batch = utils.to_cuda(X_batch)
        Y_batch = utils.to_cuda(Y_batch)

        # Perform the forward pass
        predictions = self.model(X_batch)
        # Compute the cross entropy loss for the batch
        loss = self.loss_criterion(predictions, Y_batch)
        # Backpropagation
        loss.backward()
        # Gradient descent step
        self.optimizer.step()
        # Reset all computed gradients to 0
        self.optimizer.zero_grad()
        return loss.detach().cpu().item()

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        def should_validate_model() -> bool:
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            # Perform a full pass through all the training samples
            progress = tqdm.tqdm(
                self.dataloader_train,
                desc=f"Training Epoch {epoch+1}/{self.epochs}: ",
                unit="batches",
                bar_format='{desc} |{bar:30}| {elapsed} {postfix}'
            )
            for X_batch, Y_batch in progress:
                loss = self.train_step(X_batch, Y_batch)
                self.train_history["loss"][self.global_step] = loss
                self.global_step += 1
                # Compute loss/accuracy for validation set
                if should_validate_model():
                    # Add validation loss and accuracy to progressbar
                    progress.set_postfix_str("Validating...")
                    val_loss, val_acc = self.validation_step()
                    self.save_model()
                    val = f"Validation Loss: {val_loss:.3f}, Validation Accuracy: {100*val_acc:.2f}%"
                    progress.set_postfix_str(val)

                    if self.should_early_stop():
                        progress.set_description(
                            f"Early Stopping {epoch+1}/{self.epochs}"
                        )
                        return

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            val_loss = self.val_history["loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self) -> None:
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(f"Could not load best checkpoint. {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)
