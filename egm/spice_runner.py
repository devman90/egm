import subprocess
import os
import json
import numpy as np
import random
import pickle


class SpiceRunner:
    @staticmethod
    def get_setting(title):
        config_file = os.path.join(os.path.dirname(__file__), 'configure.json')
        with open(config_file, 'r') as f:
            data = json.load(f)
            return data[title]

    def start(self, input_path):
        program = self.get_setting('ngspice')
        subprocess.call([program, input_path])


class EsdGunSpiceCreator:
    def __init__(self):
        self.spice_template = os.path.join(os.path.dirname(__file__), 'assets/esd_gun.sp')
        self.spice_complete = os.path.join(os.path.dirname(__file__), 'assets/spice_input.sp')
        self.spice_output = os.path.join(os.path.dirname(__file__), 'assets/spice_output')

    @staticmethod
    def replace_all(lines, replace_map):
        new_lines = []
        for line in lines:
            for old, new in replace_map.items():
                line = line.replace(old, new)
            new_lines.append(line)
        return new_lines

    def create(self, input_data):
        replace_map = {"RSTRAP": str(input_data['rstrap']),
                       "CSTRAP": str(input_data['cstrap']),
                       "LSTRAP": str(input_data['lstrap']),
                       "RDELAY": str(input_data['rdelay']),
                       "CDELAY": str(input_data['cdelay']),
                       "CBODY": str(input_data['cbody']),
                       "OUTPUT": self.spice_output}
        with open(self.spice_template, 'r') as rf:
            lines = rf.readlines()
            new_lines = self.replace_all(lines, replace_map)
            with open(self.spice_complete, 'w') as wf:
                for line in new_lines:
                    wf.write(line if line.endswith('\n') else (line + '\n'))
        return self.spice_complete, self.spice_output


class Waveform:
    def __init__(self, x_values, y_values):
        self.x_values = x_values
        self.y_values = y_values

    def sampled(self, sample):
        x_values = np.linspace(self.x_values[0], self.x_values[-1], sample)
        sampled_y = np.interp(x_values, self.x_values, self.y_values)
        return sampled_y


class WaveformParser:
    @staticmethod
    def parse(file_path):
        with open(file_path, 'r') as f:
            parsing_vars, parsing_values = False, False
            var = []
            values = []
            row = []

            for line in f.readlines():
                line = line.strip()
                if line == 'Variables:':
                    parsing_vars = True
                    continue
                elif parsing_vars:
                    if line == 'Values:':
                        parsing_vars = False
                        parsing_values = True
                        continue
                    elif len(line.split()) == 3:
                        if line.split()[2] == 'time':
                            var.append('x')
                        else:
                            var.append('y')

                    else:
                        print("UNKNOWN STATE")

                elif parsing_values:
                    if line == '':
                        values.append(row)
                        row = []
                    elif len(line.split()) == 2:
                        row.append(line.split()[1])
                    elif len(line.split()) == 1:
                        row.append(line.split()[0])
                    else:
                        print("UNKNOWN STATE")
            data = {v: [float(row[i]) for row in values] for i, v in enumerate(var)}
            return Waveform(data['x'], data['y'])


class EsdSimCreator:
    def __init__(self):
        self.range_map = {'rstrap': (50, 550), 'cstrap': (0, 100e-12), 'lstrap': (0, 10e-6),
                          'rdelay': (50, 550), 'cdelay': (0, 60e-12), 'cbody': (0, 600e-12)}

    def create_input_data(self):
        input_data = {}
        for key in self.range_map.keys():
            input_data[key] = random.uniform(*self.range_map[key])
        return input_data

    def normalize(self, key, value):
        if key not in self.range_map:
            print("Invalid key")
            return value
        else:
            return (value - self.range_mape[key][0]) / (self.range_mape[key][1] - self.range_map[key][0])

    def denormalize(self, key, value):
        if key not in self.range_map:
            print("Invalid key")
            return value
        else:
            return value * (self.range_mape[key][1] - self.range_map[key][0]) + self.range_mape[key][0]


class EsdSimData:
    def __init__(self):
        self.input_data = None
        self.output_data = None

    def set_input_data(self, input_data):
        self.input_data = input_data

    def set_output_data(self, output_data):
        self.output_data = output_data


class EsdSimLoader:
    @staticmethod
    def load():
        with open(os.path.join(os.path.dirname(__file__), 'assets/sim_pickle'), 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    sims = []

    loaded = EsdSimLoader.load()
    print(len(loaded))

    while True:
        input_data = EsdSimCreator().create_input_data()

        esdGun = EsdGunSpiceCreator()
        spice, out = esdGun.create(input_data=input_data)

        spiceRunner = SpiceRunner()
        spiceRunner.start(spice)

        wave = WaveformParser.parse(out)
        plt.plot(wave.sampled(10))
        plt.show()
        output_data = wave

        sim = EsdSimData()
        sim.set_input_data(input_data)
        sim.set_output_data(output_data)

        sims.append(sim)

        if len(sims) > 10:
            break
    with open(os.path.join(os.path.dirname(__file__), 'assets/sim_pickle'), 'wb') as f:
        pickle.dump(sims, f)

