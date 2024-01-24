from pathlib import Path
import csv

csv_lst = list(Path('/home/z52lu/alphasmt/ijcai24').rglob('*.csv'))
raw_lst = []

def replace_path_csv(csv_file):
    with open(csv_file, 'r') as f:
        csv_reader = csv.reader(f)
        header = next(csv_reader)
        data = list(csv_reader)
    for row in data:
        path_str = row[1]
        path = Path(path_str)
        part_to_remove = Path("/home/z52lu")
        part_to_add = Path("benchmarks")
        if path.is_symlink():
            path = path.resolve()
        path = part_to_add / path.relative_to(part_to_remove)
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

for csv_file in raw_lst:     
    replace_path_csv(csv_file)

