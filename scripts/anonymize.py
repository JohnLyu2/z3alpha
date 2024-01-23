from pathlib import Path
import csv

csv_lst = list(Path('/Users/zhengyanglumacmini/Desktop/AlphaSMT/ijcai24').rglob('*.csv'))
raw_lst = []

for csv_file in csv_lst:
    # check whether the second column in the header is 'path'
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        if header[1] == 'path':
            print(csv_file)
            raw_lst.append(csv_file)

# print(csv_lst)

