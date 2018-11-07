import sys
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from spice_runner import *


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

    def normalize(self, sim_data, sample_num):
        input_map = sim_data.input_data
        new_output = []
        for key in sorted(input_map.keys()):
            new_output.append(EsdGunSpecCreator().normalize(key, input_map[key]))

        waveform = sim_data.output_data
        sampled_waveform = waveform.sampled(sample_num)
        new_input = []
        for y in sampled_waveform:
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

    def start(self, spec):
        sample_size = spec.input_size()
        if spec.output_size() != 6:
            print("Wrong output size:", spec.output_size())
            return

        for train_sim in spec.train_data:
            x, y = EsdGunSimDataNormalizer().normalize(train_sim, sample_size)
            self.train_x.append(x)
            self.train_y.append(y)

        for val_sim in spec.validation_data:
            x, y = EsdGunSimDataNormalizer().normalize(val_sim, sample_size)
            self.validation_x.append(x)
            self.validation_y.append(y)

        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(patience=spec.early_stop_patience)
        self.train_x = np.asarray(self.train_x)
        self.train_y = np.asarray(self.train_y)
        self.validation_x = np.asarray(self.validation_x)
        self.validation_y = np.asarray(self.validation_y)
        print(self.train_x)
        print(self.train_y)
        print(self.validation_x)
        print(self.validation_y)
        hist = spec.model.fit(self.train_x, self.train_y, epochs=spec.epochs, batch_size=spec.batch_size,
                              validation_data=(self.validation_x, self.validation_y), callbacks=[early_stopping])
        self.plot_history(hist)

    def plot_history(self, hist):
        import matplotlib.pyplot as plt
        for key, val in hist.history.items():
            plt.plot(val, label=key)
        plt.legend()
        plt.show()


loaded = EsdGunSimLoader.load()
cut = len(loaded)//3
print("CUT:", cut)
train = loaded[:cut]
valid = loaded[cut:cut * 2]
test = loaded[cut * 2:]

def spec1():
    new_model = Sequential()
    new_model.add(Dense(30, input_dim=5, activation='relu'))
    new_model.add(Dense(30, activation='relu'))
    new_model.add(Dense(30, activation='relu'))
    new_model.add(Dense(30, activation='relu'))
    new_model.add(Dense(6))
    new_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    new_spec = EsdGunAgentTrainSpec(name='spec1', model=new_model, train_data=train,
                                    validation_data=valid, test_data=test)
    return new_spec

if __name__ == '__main__':
    trainer = EsdGunAgentTrainer()
    trainer.start(spec1())
