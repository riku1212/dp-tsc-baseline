import numpy as np
import tensorflow.keras as keras

from utils.utils import calculate_metrics


class ClassifierBase:

    def __init__(self, output_directory, verbose=False):
        self.output_directory = output_directory
        self.verbose = verbose
        self.callbacks = []
        self.model_path = self.output_directory + 'best_model.hdf5'

    def predict(self, x_test, y_true, return_df_metrics=True):
        model = keras.models.load_model(self.model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred

    def save_model(self):
        file_path = self.model_path
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
        return model_checkpoint