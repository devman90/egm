import sys
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from spice_runner import *
import datetime


class EsdGunAgentTrainSpec:
    def __init__(self, name, model, train_data, validation_data, test_data,
                 batch_size=128, epochs=5000, early_stop_patience=50):
        self.name = name
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop_patience = early_stop_patience

    def input_size(self):
        return self.model.input_shape[1]

    def output_size(self):
        return self.model.output_shape[1]


class EsdGunSimDataNormalizer:
    def __init__(self, wave_min=-100, wave_max=100):
        self.wave_min = wave_min
        self.wave_max = wave_max

    def normalize(self, sim_data):
        input_map = sim_data.input_data
        new_output = []
        for key in sorted(input_map.keys()):
            new_output.append(EsdGunSpecCreator().normalize(key, input_map[key]))

        waveform = sim_data.output_data.y_values
        new_input = []
        for y in waveform:
            new_input.append(self.normalize_y(y))
        return new_input, new_output

    def normalize_y(self, y):
        return (y - self.wave_min) / (self.wave_max - self.wave_min)


class EsdGunAgentTrainer:
    def __init__(self):
        self.train_x = []
        self.train_y = []
        self.validation_x = []
        self.validation_y = []
        self.test_x = []
        self.test_y = []
        self.spec = None
        self.start_time = None
        self.end_time = None
        self.save_dir_path = None

    def start(self, spec):
        self.spec = spec
        if spec.output_size() != 6:
            print("Wrong output size:", spec.output_size())
            return

        for train_sim in spec.train_data:
            x, y = EsdGunSimDataNormalizer().normalize(train_sim)
            self.train_x.append(x)
            self.train_y.append(y)

        for val_sim in spec.validation_data:
            x, y = EsdGunSimDataNormalizer().normalize(val_sim)
            self.validation_x.append(x)
            self.validation_y.append(y)

        for test_sim in spec.test_data:
            x, y = EsdGunSimDataNormalizer().normalize(test_sim)
            self.test_x.append(x)
            self.test_y.append(y)

        from keras.callbacks import EarlyStopping, CSVLogger
        early_stopping = EarlyStopping(patience=spec.early_stop_patience)
        logger = CSVLogger(os.path.join(self.save_dir(), "log.csv"))
        self.train_x = np.asarray(self.train_x)
        self.train_y = np.asarray(self.train_y)
        self.validation_x = np.asarray(self.validation_x)
        self.validation_y = np.asarray(self.validation_y)
        self.test_x = np.asanyarray(self.test_x)
        self.test_y = np.asanyarray(self.test_y)
        self.start_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        hist = spec.model.fit(self.train_x, self.train_y, epochs=spec.epochs, batch_size=spec.batch_size,
                              validation_data=(self.validation_x, self.validation_y), callbacks=[early_stopping, logger])
        self.end_time = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
        score = spec.model.evaluate(x=self.test_x, y=self.test_y)
        with open(os.path.join(self.save_dir(), 'result.txt'), 'w') as f:
            f.write(str(score) + '\n')
            f.write('start:' + self.start_time + '\n')
            f.write('end:' + self.end_time + '\n')


    def plot_history(self, hist):
        import matplotlib.pyplot as plt
        for key, val in hist.history.items():
            plt.plot(val, label=key)
        plt.legend()
        plt.show()

    def save_dir(self):
        if self.save_dir_path is None:
            str_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            path = os.path.join(os.path.dirname(__file__), 'outputs/' + self.spec.name + "_" + str_time)
            os.mkdir(path)
            self.save_dir_path = path
        return self.save_dir_path


def spec1():
    new_model = Sequential()
    new_model.add(Dense(512, input_dim=512, activation='relu'))
    new_model.add(Dense(512, activation='relu'))
    new_model.add(Dense(256, activation='relu'))
    new_model.add(Dense(256, activation='relu'))
    new_model.add(Dense(128, activation='relu'))
    new_model.add(Dense(128, activation='relu'))
    new_model.add(Dense(64, activation='relu'))
    new_model.add(Dense(64, activation='relu'))
    new_model.add(Dense(32, activation='relu'))
    new_model.add(Dense(32, activation='relu'))
    new_model.add(Dense(16, activation='relu'))
    new_model.add(Dense(16, activation='relu'))
    new_model.add(Dense(6))
    loaded = EsdGunSimLoader.load()
    cut = len(loaded) // 10
    train = loaded[:cut * 6]
    valid = loaded[cut * 6:cut * 9]
    test = loaded[cut * 9:]
    new_model.compile(loss='mse', optimizer='adam')
    new_spec = EsdGunAgentTrainSpec(name='spec1', model=new_model, train_data=train,
                                    validation_data=valid, test_data=test)
    return new_spec


def create_spec(datanum=20000, sample=256, layer_nodes=[256,128,64,32,16],
                activation='relu'):
    loaded = EsdGunSimLoader.load(datanum, sample)
    cut = len(loaded) // 10
    train = loaded[:cut * 6]
    valid = loaded[cut * 6:cut * 9]
    test = loaded[cut * 9:]

    model = Sequential()
    model.add(Dense(layer_nodes[0], input_dim=sample, activation=activation))
    for nodes in layer_nodes[1:]:
        model.add(Dense(nodes, activation=activation))
    model.add(Dense(6))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    name = 'data' + str(datanum) + '_' + 'sample' + str(sample)\
           + '_' + str(len(layer_nodes)) + 'layers' + '_' + activation
    spec = EsdGunAgentTrainSpec(name=name, model=model, train_data=train,
                                validation_data=valid, test_data=test)
    return spec


if __name__ == '__main__':
    #datanum sweep
    # EsdGunAgentTrainer().start(create_spec(datanum=100))
    # EsdGunAgentTrainer().start(create_spec(datanum=500))
    # EsdGunAgentTrainer().start(create_spec(datanum=1000))
    # EsdGunAgentTrainer().start(create_spec(datanum=5000))
    # EsdGunAgentTrainer().start(create_spec(datanum=10000))
    # EsdGunAgentTrainer().start(create_spec(datanum=50000))
    # EsdGunAgentTrainer().start(create_spec(datanum=100000))
    for _ in range(3):
        # Sampling sweep
        EsdGunAgentTrainer().start(create_spec(sample=16))
        EsdGunAgentTrainer().start(create_spec(sample=32))
        EsdGunAgentTrainer().start(create_spec(sample=64))
        EsdGunAgentTrainer().start(create_spec(sample=128))
        EsdGunAgentTrainer().start(create_spec(sample=256))
        EsdGunAgentTrainer().start(create_spec(sample=512))
        EsdGunAgentTrainer().start(create_spec(sample=1024))

    for _ in range(3):
        # Layer sweep
        EsdGunAgentTrainer().start(create_spec(layer_nodes=[128]))
        EsdGunAgentTrainer().start(create_spec(layer_nodes=[128, 64, 32]))
        EsdGunAgentTrainer().start(create_spec(layer_nodes=[128, 128, 64, 64, 32, 32]))
        EsdGunAgentTrainer().start(create_spec(layer_nodes=[256, 256, 128, 128, 64, 64, 32, 32]))

    for _ in range(3):
        # Activation sweep
        EsdGunAgentTrainer().start(create_spec(activation='tanh'))
