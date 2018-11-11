import os
import datetime


class ReportGenerator:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
        self.save_path = os.path.join(os.path.dirname(__file__), 'result.csv')

    def start(self):
        file_names = os.listdir(self.output_dir)
        for file_name in file_names:
            abs_file_name = os.path.join(self.output_dir, file_name)
            self.parse_dir(abs_file_name)

    def parse_dir(self, dir_name):
        basename = os.path.basename(dir_name)
        items = basename.split('_')[:4]
        items += self.parse_log(os.path.join(dir_name, 'result.txt'))
        print(','.join(items))

    def parse_log(self, log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            loss = lines[0].split(',')[0].replace('[', '')
            start_time = datetime.datetime.strptime(lines[1].split(":")[1].strip(), '%y%m%d_%H%M%S')
            end_time = datetime.datetime.strptime(lines[2].split(':')[1].strip(), '%y%m%d_%H%M%S')
            return [loss, str((end_time - start_time).total_seconds())]
        return ["", ""]


if __name__ == '__main__':
    ReportGenerator().start()
