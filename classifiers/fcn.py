# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
import time

import numpy as np
import tensorflow as tf

from classifiers.base import ClassifierBase, ComputeRDP, compute_dp_sgd_privacy
from utils.utils import save_logs

from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer

NUMBER_OF_EPOCHS = 100  # 2000
BATCH_SIZE = 60
NOISE_MULTIPLIER = 1.1

class ClassifierFCN(ClassifierBase):

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        super().__init__(output_directory, verbose)
        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()

            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        input_layer = tf.keras.layers.Input(input_shape)

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

        output_layer = tf.keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        # loss = tf.tf.keras.losses.CategoricalCrossentropy()
        # optimizer = tf.tf.keras.optimizers.Adam()

        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
        optimizer = DPAdamGaussianOptimizer(noise_multiplier=NOISE_MULTIPLIER,
                                            l2_norm_clip=1.0,
                                            num_microbatches=60,
                                            learning_rate=0.15)

        model.compile(loss=loss, optimizer=optimizer,
                      metrics=['accuracy'])

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

        start_time = time.time()

        self.callbacks = [ComputeRDP(BATCH_SIZE, len(x_train), NOISE_MULTIPLIER)]

        # note that we ignore batch_size as it may not divide input
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
