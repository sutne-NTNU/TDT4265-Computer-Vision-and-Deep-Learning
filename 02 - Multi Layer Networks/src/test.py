import unittest
import numpy as np

import utils
from task2ab import SoftmaxModel, one_hot_encode, pre_process_images, cross_entropy_loss
from task2c import get_predictions


"""
This file contains all the assertions found in the original files and have been
placed in their own unit tests based on what they are testing, the unit test are
in other words testing exactly the same functionality but they are all now located
in this file for a better overview over what is working and what isn't,
less duplicate code, and less clutter in the task files.
"""


X_raw, Y_raw, *_ = utils.load_full_mnist()
X_train = pre_process_images(X_raw)
Y_train = one_hot_encode(Y_raw, 10)


class Task2a(unittest.TestCase):

    def test_one_hot_encode(self):
        y = np.zeros((1, 1), dtype=int)
        y[0, 0] = 3
        y = one_hot_encode(y, 10)
        self.assertTrue(
            y[0, 3] == 1 and y.sum() == 1,
            msg=f'Expected [0,0,0,1,0,0,0,0,0,0], but got {y}'
        )

    def test_cross_entropy_loss(self):
        target = np.zeros((2, 3))
        target[0, 1] = 1
        target[1, 2] = 1
        output = np.array([[0.1, 0.8, 0.1],
                           [0.1, 0.8, 0.1]])
        loss = cross_entropy_loss(target, output)
        self.assertAlmostEqual(loss, 1.2628643)

    def test_pre_process_images(self):
        processed = pre_process_images(X_raw, verbose=True)
        self.assertEqual(
            processed.shape[1], 785,
            msg=f'Expected X_train to have 785 elements per image. Shape was: {processed.shape}'
        )


class Task2b(unittest.TestCase):

    def test_model_backward_grad_shape(self):
        model = SoftmaxModel(neurons_per_layer=[64, 10])
        out = model.forward(X_train)
        model.backward(X_train, out, Y_train)
        for grad, w in zip(model.grads, model.weights):
            self.assertTrue(
                grad.shape == w.shape,
                msg=f'Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}.'
            )


def gradient_approximation(model: SoftmaxModel) -> tuple:
    ''' Numerical approximation for gradients. Should not be edited.
        Details about this test is given in the appendix in the assignment. '''
    X = X_train[:100]
    Y = Y_train[:100]
    for layer_idx, w in enumerate(model.weights):
        model.weights[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    epsilon = 1e-3
    for layer_idx, w in enumerate(model.weights):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.weights[layer_idx][i, j].copy()
                model.weights[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.weights[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.weights[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                actual = model.grads[layer_idx][i, j]
                difference = gradient_approximation - actual
                if abs(difference) >= epsilon**2:
                    return False, f'''Calculated gradient is incorrect.
                    Layer IDX = {layer_idx}, i={i}, j={j}.
                    Approximation:   {gradient_approximation}
                    actual gradient: {actual}
                    If this test fails there could be errors in your cross
                    entropy loss function, forward function or backward function'''
    return True, None


class Task2c(unittest.TestCase):

    def test_get_prediction(self):
        output = np.array([[0.1, 0.8, 0.1],
                           [0.4, 0.4, 0.1]])
        predictions = get_predictions(output)
        expected = np.array([[0, 1, 0],
                             [1, 0, 0]])
        self.assertTrue(np.array_equal(predictions, expected))

    def test_single_hidden_layer(self):
        model = SoftmaxModel(neurons_per_layer=[64, 10])
        success, msg = gradient_approximation(model)
        self.assertTrue(success, msg=msg)


class Task4c(unittest.TestCase):

    def test_multi_hidden_layer(self):
        model = SoftmaxModel(
            neurons_per_layer=[64, 64, 10],
            use_improved_weight_init=True,
            use_improved_sigmoid=True,
        )
        success, msg = gradient_approximation(model)
        self.assertTrue(success, msg=msg)


if __name__ == '__main__':
    unittest.main()
