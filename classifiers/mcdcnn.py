# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import time

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

from classifiers.base import ClassifierBase
from utils.utils import save_logs

BATCH_SIZE = 16
NUMBER_OF_EPOCHS = 10 #120


class ClassifierMCDCNN(ClassifierBase):

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        super().__init__(output_directory, verbose)
        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()

            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        n_t = input_shape[0]
        n_vars = input_shape[1]

        padding = 'valid'

        if n_t < 60: # for ItalyPowerOndemand
            padding = 'same'

        input_layers = []
        conv2_layers = []

        for n_var in range(n_vars):
            input_layer = keras.layers.Input((n_t,1))
            input_layers.append(input_layer)

            conv1_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(input_layer)
            conv1_layer = keras.layers.MaxPooling1D(pool_size=2)(conv1_layer)

            conv2_layer = keras.layers.Conv1D(filters=8,kernel_size=5,activation='relu',padding=padding)(conv1_layer)
            conv2_layer = keras.layers.MaxPooling1D(pool_size=2)(conv2_layer)
            conv2_layer = keras.layers.Flatten()(conv2_layer)

            conv2_layers.append(conv2_layer)

        if n_vars == 1:
            # to work with univariate time series
            concat_layer = conv2_layers[0]
        else:
            concat_layer = keras.layers.Concatenate(axis=-1)(conv2_layers)

        fully_connected = keras.layers.Dense(units=732,activation='relu')(concat_layer)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(fully_connected)

        model = keras.models.Model(inputs=input_layers, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=0.0005),
                      metrics=['accuracy'])

        self.callbacks = [self.save_model()]

        return model

    def prepare_input(self,x):
        new_x = []
        n_t = x.shape[1]
        n_vars = x.shape[2]

        for i in range(n_vars):
            new_x.append(x[:,:,i:i+1])

        return  new_x

    def fit(self, x, y, x_test, y_test, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        mini_batch_size = BATCH_SIZE
        nb_epochs = NUMBER_OF_EPOCHS

        x_train, x_val, y_train, y_val = \
            train_test_split(x, y, test_size=0.33)

        x_test = self.prepare_input(x_test)
        x_train = self.prepare_input(x_train)
        x_val = self.prepare_input(x_val)

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory+'last_model.hdf5')

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5')

        y_pred = model.predict(x_test)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration,lr=False)

        keras.backend.clear_session()
