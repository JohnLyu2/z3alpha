from pathlib import Path
import csv

csv_lst = list(Path('/Users/zhengyanglumacmini/Desktop/AlphaSMT/ijcai24').rglob('*.csv'))
raw_lst = []

def replace_symbolink_csv(csv_file):
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        data = list(csv_reader)
    for row in data:
        path_str = row[1]
        path = Path(path_str)
        if path.is_symlink():
            path = path.resolve()
            row[1] = str(path)
    with open(csv_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        csv_writer.writerows(data)

for csv_file in csv_lst:
    # check whether the second column in the header is 'path'
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        if header[1] == 'path':
            print(csv_file)
            raw_lst.append(csv_file)

# print(csv_lst)

