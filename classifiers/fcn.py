# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import time

import numpy as np
import tensorflow as tf

from classifiers.base import ClassifierBase, ComputeRDP, compute_dp_sgd_privacy
from utils.utils import save_logs
from pathlib import PureWindowsPath

from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer

NUMBER_OF_EPOCHS = 1000  # 2000
BATCH_SIZE = 32
NOISE_MULTIPLIER = 3 # note that this affects eps and accuracy.
# if noise multiplier is small, accuracy increases but epsilon also. if noise increases, eps decreases but accuracy
# drops


class ClassifierFCN(ClassifierBase):

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, threshold=10):
        super().__init__(output_directory, verbose, threshold)
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        if build:
            self.model_sgd = self.build_model()
            self.model = self.build_model(sgd=False)
            self.model_sgd_path = PureWindowsPath(self.output_directory) / 'dp'
            self.model_path = PureWindowsPath(self.output_directory) / 'normal'
            if verbose:
                self.model_sgd.summary()
                self.model.summary()

            self.model_sgd.save_weights(str(self.model_sgd_path / 'model_init.hdf5'))
            self.model.save_weights(str(self.model_path / 'model_init.hdf5'))

    def build_model(self, sgd=True):
        input_layer = tf.keras.layers.Input(self.input_shape)

        conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)

        conv3 = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Activation('relu')(conv3)

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = tf.keras.layers.Dense(self.nb_classes, activation='softmax')(gap_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        if sgd:
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
            optimizer = DPAdamGaussianOptimizer(noise_multiplier=NOISE_MULTIPLIER,
                                                l2_norm_clip=0.5,
                                                num_microbatches=BATCH_SIZE,
                                                learning_rate=0.0001)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy()
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        #           monitor='loss',
        #           factor=0.5,
        #           patience=50,
        #           vmin_lr=0.0001
        #          )

        # self.callbacks = [reduce_lr, self.save_model()]
        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        # for DP
        self.callbacks = [ComputeRDP(BATCH_SIZE, len(x_train), NOISE_MULTIPLIER, self.threshold),
                          self.save_model(str(self.model_sgd_path / 'best_model.hdf5'))]

        start_time = time.time()
        hist = self.model_sgd.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        duration = time.time() - start_time

        self.model_sgd.save(str(self.model_sgd_path / 'last_model.hdf5'))

        model = tf.keras.models.load_model(str(self.model_sgd_path / 'best_model.hdf5'))

        y_pred = model.predict(x_val)
        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.model_sgd_path, hist, y_pred, y_true, duration, lr=False)

        # non-DP
        stopped_epoch = len(hist.epoch)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.00001)
        self.callbacks = [reduce_lr, self.save_model(str(self.model_path / 'best_model.hdf5'))]

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
