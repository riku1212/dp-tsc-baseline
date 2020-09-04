import numpy as np
import tensorflow as tf

from utils.utils import calculate_metrics
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp  # pylint: disable=g-import-not-at-top
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
import math
import pandas as pd

class ClassifierBase:

    def __init__(self, output_directory, verbose=False, threshold=10):
        self.output_directory = output_directory
        self.verbose = verbose
        self.callbacks = []
        self.threshold = threshold

    def predict(self, x_test, y_true, model_path, return_df_metrics=True):
        model = tf.keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred

    def save_model(self, model_path):
        file_path = model_path
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        return model_checkpoint


def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
    """Compute and print results of DP-SGD analysis."""

    # compute_rdp requires that sigma be the ratio of the standard deviation of
    # the Gaussian noise to the l2-sensitivity of the function to which it is
    # added. Hence, sigma here corresponds to the `noise_multiplier` parameter
    # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
    rdp = compute_rdp(q, sigma, steps, orders)

    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)

    print('DP-SGD with sampling rate = {:.3g}% and noise_multiplier = {} iterated'
          ' over {} steps satisfies'.format(100 * q, sigma, steps), end=' ')
    print('differential privacy with eps = {:.3g} and delta = {}.'.format(
        eps, delta))
    print('The optimal RDP order is {}.'.format(opt_order))

    if opt_order == max(orders) or opt_order == min(orders):
        print('The privacy estimate is likely to be improved by expanding '
              'the set of orders.')

    return eps, opt_order


def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):
    """Compute epsilon based on the given hyperparameters."""
    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        raise app.UsageError('n must be larger than the batch size.')
    orders = ([0.25 * x for x in range(8, 81)] +
              list(range(21, 65)) + [128, 256, 512])
    steps = int(math.ceil(epochs * n / batch_size))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)


class ComputeRDP(tf.keras.callbacks.Callback):

    def __init__(self, batch_size, input_size, noise_multiplier, threshold, model_path):
        self.batch_size = batch_size
        self.input_size = input_size
        self.noise_multiplier = noise_multiplier
        self.threshold = threshold
        self.model_path = model_path
        self.eps_table = pd.DataFrame(columns=['epoch', 'eps', 'train_acc', 'val_acc'])

    def on_epoch_end(self, epoch, logs=None):
        delta = min(1/self.input_size, 0.0000001)
        eps, _ = compute_dp_sgd_privacy(
            self.input_size, self.batch_size, self.noise_multiplier, epoch, delta)
        self.eps_table = self.eps_table.append({
            'epoch': epoch,
            'eps': eps,
            'train_acc': logs['accuracy'],
            'val_acc': logs['val_accuracy']
        }, ignore_index=True)

        print('\nFor delta={delta}, the current epsilon is: {eps}'.format(delta=delta, eps=eps))
        if eps > self.threshold:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.eps_table.to_csv(self.model_path / "eps_history.csv", index=False)
