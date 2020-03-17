import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from classifiers.base import ClassifierBase
from utils.utils import save_logs

NUMBER_OF_EPOCHS = 2000
BATCH_SIZE = 16


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
        input_layer = keras.layers.Input(shape=input_shape)
        input_layer_masked = keras.layers.Masking()(input_layer)

        layer_1 = keras.layers.LSTM(units=3,
                                    input_shape=(input_shape),
                                    return_sequences=True,
                                    activation='tanh'
                                    )(input_layer_masked)

        output_layer = keras.layers.LSTM(units=nb_classes,
                 activation='tanh',
                 return_sequences=False)(layer_1)
        output_layer = keras.layers.Dropout(0.2)(output_layer)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=200, min_lr=0.1)

        self.callbacks = [reduce_lr, self.save_model()]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = BATCH_SIZE
        nb_epochs = NUMBER_OF_EPOCHS

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()
