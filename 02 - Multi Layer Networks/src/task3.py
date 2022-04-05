import matplotlib.pyplot as plt

import utils
from task2ab import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2c import SoftmaxTrainer, calculate_accuracy


results_file = 'results/task3_final_results.txt'


def write_results_to_file(description: str, model: SoftmaxModel):
    t_loss = cross_entropy_loss(Y_train, model.forward(X_train))
    v_loss = cross_entropy_loss(Y_val, model.forward(X_val))
    t_acc = calculate_accuracy(X_train, Y_train, model)
    v_acc = calculate_accuracy(X_val, Y_val, model)
    utils.write_results_to_file(
        results_file, description, t_loss, v_loss, t_acc, v_acc
    )


def train_and_return_model_and_history(
    use_improved_weight_init: bool = False,
    use_improved_sigmoid: bool = False,
    momentum_gamma: float = 0.0,  # 0 => use_momentum = False by default
    learning_rate: float = .1,
):
    """ 
    This method just prevents duplicate code and makes it easier 
    to see what attributes are changed for each subtask.
    """
    model = SoftmaxModel(
        neurons_per_layer=[64, 10],
        use_improved_weight_init=use_improved_weight_init,
        use_improved_sigmoid=use_improved_sigmoid,
    )
    trainer = SoftmaxTrainer(
        model=model,
        X_train=X_train,
        X_val=X_val,
        Y_train=Y_train,
        Y_val=Y_val,
        batch_size=32,
        shuffle_dataset=True,
        learning_rate=learning_rate,
        momentum_gamma=momentum_gamma,
    )
    training, validation = trainer.train(num_epochs=50)
    return model, training, validation


if __name__ == '__main__':
    utils.setup(results_file)

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train, X_val = pre_process_images(X_train), pre_process_images(X_val)
    Y_train, Y_val = one_hot_encode(Y_train, 10), one_hot_encode(Y_val, 10)

    # Create and Train model without any modifications (Same as Task 2C)
    desc_norm = 'No Improvements'
    model, train, val = train_and_return_model_and_history()

    # Task 3a
    desc_3a = 'Improved Weight Initialization'
    model_3a, train_3a, val_3a = train_and_return_model_and_history(
        use_improved_weight_init=True,  # Adding Improved Weight Init
    )

    # Task 3b
    desc_3b = 'Improved Sigmoid'
    model_3b, train_3b, val_3b = train_and_return_model_and_history(
        use_improved_sigmoid=True,  # Adding Improved Sigmoid
    )

    # Task 3c
    desc_3c = 'Using Momentum'
    model_3c, train_3c, val_3c = train_and_return_model_and_history(
        momentum_gamma=0.9,  # Adding Momentum
        learning_rate=.02,  # Reducing Learning Rate
    )

    # Compare each models final loss and accuracies in a table (formatted in a file)
    utils.write_markdown_header(results_file)
    write_results_to_file(desc_norm, model)
    write_results_to_file(desc_3a, model_3a)
    write_results_to_file(desc_3b, model_3b)
    write_results_to_file(desc_3c, model_3c)

    # Plot Loss
    plt.figure(figsize=(16, 10))
    ylim = (0.0, 0.5)
    # Training
    plt.subplot(1, 2, 1)
    plt.ylim(ylim)
    plt.title('Training Loss For The Different Models')
    utils.plot_loss(train['loss'], desc_norm, 100)
    utils.plot_loss(train_3a['loss'], desc_3a, 100)
    utils.plot_loss(train_3b['loss'], desc_3b, 100)
    utils.plot_loss(train_3c['loss'], desc_3c, 100)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Training Loss')
    plt.legend(loc='upper right')
    # Validation
    plt.subplot(1, 2, 2)
    plt.ylim(ylim)
    plt.title('Validation Loss For The Different Models')
    utils.plot_loss(val['loss'], desc_norm)
    utils.plot_loss(val_3a['loss'], desc_3a)
    utils.plot_loss(val_3b['loss'], desc_3b)
    utils.plot_loss(val_3c['loss'], desc_3c)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('results/task3_loss.png')

    # Plot Accuracies
    plt.figure(figsize=(16, 10))
    ylim = (0.85, 1.005)
    # Training
    plt.subplot(1, 2, 1)
    plt.ylim(ylim)
    plt.title('Training Accuracy For The Different Models')
    utils.plot_loss(train['accuracy'], desc_norm)
    utils.plot_loss(train_3a['accuracy'], desc_3a)
    utils.plot_loss(train_3b['accuracy'], desc_3b)
    utils.plot_loss(train_3c['accuracy'], desc_3c)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Training Accuracy')
    plt.legend(loc='lower right')
    # Validation
    plt.subplot(1, 2, 2)
    plt.ylim(ylim)
    plt.title('Validation Accuracy For The Different Models')
    utils.plot_loss(val['accuracy'], desc_norm)
    utils.plot_loss(val_3a['accuracy'], desc_3a)
    utils.plot_loss(val_3b['accuracy'], desc_3b)
    utils.plot_loss(val_3c['accuracy'], desc_3c)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Validation Accuracy')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('results/task3_accuracy.png')
