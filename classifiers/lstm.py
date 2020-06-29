import time

import numpy as np
import tensorflow as tf

from classifiers.base import ClassifierBase, ComputeRDP, compute_dp_sgd_privacy
from utils.utils import save_logs
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer

NUMBER_OF_EPOCHS = 100
BATCH_SIZE = 60
NOISE_MULTIPLIER = 1.1


class ClassifierLSTM(ClassifierBase):

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        super().__init__(output_directory, verbose)
        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()

            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        print(input_shape)
        # model = tf.keras.Sequential([
        #     tf.keras.layers.Input(shape=input_shape),
        #     tf.keras.layers.Masking(),
        #     tf.keras.layers.LSTM(units=128, activation='tanh'),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(nb_classes, activation='softmax')
        # ])
        input_layer = tf.keras.layers.Input(shape=input_shape)
        input_layer_masked = tf.keras.layers.Masking()(input_layer)

        layer_1 = tf.keras.layers.LSTM(units=3,
                                       input_shape=(input_shape),
                                       return_sequences=True,
                                       activation='tanh'
                                       )(input_layer_masked)

        output_layer = tf.keras.layers.LSTM(units=nb_classes,
                                            activation='tanh',
                                            return_sequences=False)(layer_1)

        output_layer = tf.keras.layers.Dropout(0.2)(output_layer)
        output_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
        optimizer = DPAdamGaussianOptimizer(noise_multiplier=NOISE_MULTIPLIER,
                                            l2_norm_clip=1.0,
                                            num_microbatches=60,
                                            learning_rate=0.15)
        model.compile(loss=loss, optimizer=optimizer,
                      metrics=['accuracy'])

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)
        #
        # self.callbacks = [reduce_lr, self.save_model()]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        # mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        self.callbacks = [ComputeRDP(BATCH_SIZE, len(x_train), NOISE_MULTIPLIER)]

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUMBER_OF_EPOCHS,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = tf.keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        tf.keras.backend.clear_session()
