import time

import numpy as np
import tensorflow as tf

from classifiers.base import ClassifierBase, ComputeRDP
from utils.utils import save_logs
from pathlib import PureWindowsPath
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer

NUMBER_OF_EPOCHS = 2000
BATCH_SIZE = 4
NOISE_MULTIPLIER = 3
LEARNING_RATE = 0.01
NORM_CLIP = 0.5


class ClassifierLSTM(ClassifierBase):

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, threshold=10):
        super().__init__(output_directory, verbose, threshold)
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        if build:
            self.model_dp = self.build_model()
            self.model = self.build_model(sgd=False)
            self.model_dp_path = PureWindowsPath(self.output_directory) / 'dp'
            self.model_path = PureWindowsPath(self.output_directory) / 'normal'
            if verbose:
                self.model_dp.summary()
                self.model.summary()

            self.model_dp.save_weights(str(self.model_dp_path / 'model_init.hdf5'))
            self.model.save_weights(str(self.model_path / 'model_init.hdf5'))

    def build_model(self, sgd=True):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        output_layer = tf.keras.layers.LSTM(units=200,
                                            activation='tanh',
                                            return_sequences=False)(input_layer)

        output_layer = tf.keras.layers.Dropout(0.2)(output_layer)
        output_layer = tf.keras.layers.Dense(self.nb_classes, activation='softmax')(output_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        if sgd:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
            optimizer = DPAdamGaussianOptimizer(noise_multiplier=NOISE_MULTIPLIER,
                                                l2_norm_clip=NORM_CLIP,
                                                num_microbatches=BATCH_SIZE,
                                                learning_rate=LEARNING_RATE)
        else:
            loss = 'categorical_crossentropy'
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        # for DP
        self.callbacks = [ComputeRDP(BATCH_SIZE, len(x_train), NOISE_MULTIPLIER, self.threshold, self.model_dp_path),
                          self.save_model(str(self.model_dp_path / 'best_model.hdf5'))]

        start_time = time.time()
        hist = self.model_dp.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS,
                                 verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        duration = time.time() - start_time

        self.model_dp.save(str(self.model_dp_path / 'last_model.hdf5'))

        model = tf.keras.models.load_model(str(self.model_dp_path / 'best_model.hdf5'))

        y_pred = model.predict(x_val)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.model_dp_path, hist, y_pred, y_true, duration, lr=False)

        # non-DP
        stopped_epoch = len(hist.epoch)
        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.00001)
        self.callbacks = [self.save_model(str(self.model_path / 'best_model.hdf5'))]

        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=stopped_epoch,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        duration = time.time() - start_time

        self.model.save(str(self.model_path / 'last_model.hdf5'))

        model = tf.keras.models.load_model(str(self.model_path / 'best_model.hdf5'))

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.model_path, hist, y_pred, y_true, duration, lr=False)

        tf.keras.backend.clear_session()
