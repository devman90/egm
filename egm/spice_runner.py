import subprocess
import tempfile
import os
import json
import numpy as np
import pandas as pd


def replace_all(lines, replace_map):
    new_lines = []
    for line in lines:
        for old, new in replace_map.items():
            line = line.replace(old, new)
        new_lines.append(line)
    return new_lines


def manipulate_file(input_file, output_file, replace_map):
    with open(input_file, 'r') as rf:
        lines = rf.readlines()
        new_lines = replace_all(lines, replace_map)
        with open(output_file, 'w') as wf:
            for line in new_lines:
                wf.write(line if line.endswith('\n') else (line + '\n'))


def parse_spice_output(file_path):
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
        return data


def get_waveform(rstrap, cstrap, lstrap, rdelay, cdelay, cbody, resolution=300):
    template_deck = os.path.join(os.path.dirname(__file__), 'assets/esd_gun.sp')
    (_, spice_out) = tempfile.mkstemp()
    replace_map = {"RSTRAP": str(rstrap), "CSTRAP": str(cstrap), "LSTRAP": str(lstrap),
                   "RDELAY": str(rdelay), "CDELAY": str(cdelay), "CBODY": str(cbody), "OUTPUT": spice_out}
    (_, temp_deck) = tempfile.mkstemp()
    manipulate_file(template_deck, temp_deck, replace_map)
    #print(spice_out, temp_deck)

    configure_json = os.path.join(os.path.dirname(__file__), 'configure.json')
    with open(configure_json, 'r') as f:
        data = json.load(f)
        #print(data['ngspice'])
        subprocess.call([data['ngspice'], temp_deck])
        wave = parse_spice_output(spice_out)
        x_values = np.linspace(wave['x'][0], wave['x'][-1], num=resolution)
        y_values = np.interp(x_values, wave['x'], wave['y'])
        return y_values


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    while True:
        rstrap = random.uniform(50, 550)
        cstrap = random.uniform(0.0, 100e-12)
        lstrap = random.uniform(0, 10e-6)
        rdelay = random.uniform(50, 550)
        cdelay = random.uniform(0, 60e-12)
        cbody = random.uniform(0, 600e-12)
        print(rstrap, cstrap, lstrap, rdelay, cdelay, cbody)
        # wave = get_waveform(287, 8.12e-11, 2.79e-08, 285, 3.21e-11, 1.8e-10)
        wave = get_waveform(rstrap=rstrap, cstrap=cstrap, lstrap=lstrap, rdelay=rdelay,
                            cdelay=cdelay, cbody=cbody)

        resolution = 300
        x_values = np.linspace(wave['x'][0], wave['x'][-1], num=resolution)
        y_values = np.interp(x_values, wave['x'], wave['y'])

        print(x_values)
        plt.plot(x_values, y_values)
        plt.show()


