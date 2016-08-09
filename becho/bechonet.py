from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.callbacks import Callback
import csv


class BechoNet(object):

    def __init__(self, **kwargs):
        self.verbose = kwargs.get('verbose', False)
        self.nodes_1 = kwargs.get('nodes_1', 164)
        self.nodes_2 = kwargs.get('nodes_2', 150)
        self.num_actions = kwargs.get('num_actions')
        self.num_inputs = kwargs.get('num_inputs')

        self.optimizer = kwargs.get('optimizer', 'rmsprop')
        self.activation = kwargs.get('activation', 'relu')
        self.learning_rate = kwargs.get('learning_rate', 0.0001)

        self.load_weights = kwargs.get('load_weights', False)
        self.save_weights = kwargs.get('save_weights', False)

        self.log_file = kwargs.get('log_file', 'losslog.txt')
        self.weights_file = kwargs.get('weights_file', None)
        self.loss_log = []

        self.model = self.nn()

        if self.verbose:
            print(
                """
                    Creating neural net with options:
                    Nodes, layer 1: %d
                    Nodes, layer 2: %d
                    Load weights?: %s
                    Save weights?: %s
                    Weights file: %s
                    Log file: %s
                """
                % (self.nodes_1,
                   self.nodes_2,
                   str(self.load_weights),
                   str(self.save_weights),
                   self.weights_file,
                   self.log_file)
            )

    def nn(self):
        model = Sequential()

        # First layer.
        model.add(Dense(
            self.nodes_1, init='lecun_uniform',
            input_shape=(self.num_inputs,)
        ))
        model.add(Activation(self.activation))

        # Second layer.
        model.add(Dense(self.nodes_2, init='lecun_uniform'))
        model.add(Activation(self.activation))

        # Output layer.
        model.add(Dense(self.num_actions, init='lecun_uniform'))
        model.add(Activation('linear'))

        if self.optimizer == 'rmsprop':
            optimizer = RMSprop(self.learning_rate)
        elif self.optimizer == 'adam':
            optimizer = Adam(self.learning_rate)
        elif self.optimizer == 'adadelta':
            optimizer = Adadelta(self.learning_rate)
        else:
            print('no optimizer')
        model.compile(loss='mse', optimizer=optimizer)

        if self.load_weights and self.weights_file is not None:
            if self.verbose:
                print("Loading weights from %s" % self.weights_file)
            model.load_weights(self.weights_file)

        return model

    def predict(self, state):
        try:
            return self.model.predict(state.reshape(1, self.num_inputs),
                                      batch_size=1)
        except:
            print(state)
            raise Exception("PredictionError")

    def train(self, X, y, batch_size):
        history = LossHistory()
        self.model.fit(
            X, y, batch_size=batch_size,
            nb_epoch=1, verbose=0, callbacks=[history]
        )
        self.loss_log.append(history.losses)

    def log_results(self):
        # Save the results to a file so we can graph it later.
        with open(self.log_file, 'w') as lf:
            wr = csv.writer(lf)
            for loss_item in self.loss_log:
                wr.writerow(loss_item)
        self.loss_log = []

    def save_weights_file(self):
        if self.weights_file is not None:
            self.model.save_weights(self.weights_file, overwrite=True)


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
