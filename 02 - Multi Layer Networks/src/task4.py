import matplotlib.pyplot as plt

import utils
from task2ab import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2c import SoftmaxTrainer, calculate_accuracy


results_file = 'results/task4_final_results.txt'


def write_results_to_file(description: str, model: SoftmaxModel):
    t_loss = cross_entropy_loss(Y_train, model.forward(X_train))
    v_loss = cross_entropy_loss(Y_val, model.forward(X_val))
    t_acc = calculate_accuracy(X_train, Y_train, model)
    v_acc = calculate_accuracy(X_val, Y_val, model)
    utils.write_results_to_file(
        results_file, description, t_loss, v_loss, t_acc, v_acc
    )


def train_and_return_model_and_history(neurons_per_layer: list):
    model = SoftmaxModel(
        neurons_per_layer=neurons_per_layer,
        use_improved_weight_init=True,  # Improved Weight Initialization
        use_improved_sigmoid=True,  # Improved Sigmoid
    )
    trainer = SoftmaxTrainer(
        model=model,
        X_train=X_train,
        X_val=X_val,
        Y_train=Y_train,
        Y_val=Y_val,
        batch_size=32,
        shuffle_dataset=True,
        learning_rate=.02,  # Reduced Learning Rate
        momentum_gamma=0.9,  # Momentum
    )
    training, validation = trainer.train(num_epochs=50)
    return model, training, validation


if __name__ == "__main__":
    utils.setup(results_file)

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train, X_val = pre_process_images(X_train), pre_process_images(X_val)
    Y_train, Y_val = one_hot_encode(Y_train, 10), one_hot_encode(Y_val, 10)

   # Create and Train model without any modifications (Same as Task 2C)
    desc_norm = 'Hidden Layer with 64 Neurons'
    model, train, val = train_and_return_model_and_history(
        # 785*64 + 64*10 = 50 880 parameters
        neurons_per_layer=[64, 10]
    )

    # Task 4a
    desc_4a = 'Hidden Layer With 32 Neurons'
    model_4a, train_4a, val_4a = train_and_return_model_and_history(
        neurons_per_layer=[32, 10]
    )

    # Task 4b
    desc_4b = 'Hidden Layer With 128 Neurons'
    model_4b, train_4b, val_4b = train_and_return_model_and_history(
        neurons_per_layer=[128, 10]
    )

    # Task 4d
    desc_4d = '2 Hidden Layers With 60 Neurons'
    model_4d, train_4d, val_4d = train_and_return_model_and_history(
        # (784+1)*60 + 60*60 + 60*10 = 51 300 paramters
        neurons_per_layer=[60, 60, 10]
    )

    # Task 4e
    desc_4e = '10 Hidden Layers With 64 Neurons'
    model_4e, train_4e, val_4e = train_and_return_model_and_history(
        neurons_per_layer=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]
    )

    # Compare each models final loss and accuracies in a table (formatted for markdown in a file)
    utils.write_markdown_header(results_file)
    write_results_to_file(desc_norm, model)
    write_results_to_file(desc_4a, model_4a)
    write_results_to_file(desc_4b, model_4b)
    write_results_to_file(desc_4d, model_4d)
    write_results_to_file(desc_4e, model_4e)

    # Plot Loss
    plt.figure(figsize=(16, 10))
    ylim = (0.0, 0.5)
    # Training
    plt.subplot(1, 2, 1)
    plt.ylim(ylim)
    plt.title('Training Loss For The Different Models')
    utils.plot_loss(train['loss'], desc_norm, 100)
    utils.plot_loss(train_4a['loss'], desc_4a, 100)
    utils.plot_loss(train_4b['loss'], desc_4b, 100)
    utils.plot_loss(train_4d['loss'], desc_4d, 100)
    utils.plot_loss(train_4e['loss'], desc_4e, 100)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Training Loss')
    plt.legend(loc='upper right')
    # Validation
    plt.subplot(1, 2, 2)
    plt.ylim(ylim)
    plt.title('Validation Loss For The Different Models')
    utils.plot_loss(val['loss'], desc_norm)
    utils.plot_loss(val_4a['loss'], desc_4a)
    utils.plot_loss(val_4b['loss'], desc_4b)
    utils.plot_loss(val_4d['loss'], desc_4d)
    utils.plot_loss(val_4e['loss'], desc_4e)
    plt.xlabel('Number of Training Steps')
    plt.ylabel('Validation Loss')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("results/task4_loss.png")

    # Plot Accuracies
    plt.figure(figsize=(16, 10))
    ylim = (0.92, 1.005)
    # Training
    plt.subplot(1, 2, 1)
    plt.ylim(ylim)
    plt.title('Training Accuracy For The Different Models')
    utils.plot_loss(train["accuracy"], desc_norm)
    utils.plot_loss(train_4a["accuracy"], desc_4a)
    utils.plot_loss(train_4b["accuracy"], desc_4b)
    utils.plot_loss(train_4d["accuracy"], desc_4d)
    utils.plot_loss(train_4e["accuracy"], desc_4e)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='lower right')
    # Validation
    plt.subplot(1, 2, 2)
    plt.ylim(ylim)
    plt.title('Validation Accuracy For The Different Models')
    utils.plot_loss(val["accuracy"], desc_norm)
    utils.plot_loss(val_4a["accuracy"], desc_4a)
    utils.plot_loss(val_4b["accuracy"], desc_4b)
    utils.plot_loss(val_4d["accuracy"], desc_4d)
    utils.plot_loss(val_4e["accuracy"], desc_4e)
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig("results/task4_accuracy.png")
