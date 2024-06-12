import csv

TARGET_CSV = "/home/z52lu/z3alpha/smtcomp24/results/QF_SNIA/z3str.csv"

COMPARE_CSV_LST = [
    "/home/z52lu/z3alpha/smtcomp24/results/QF_SNIA/cvc5.csv"
]

# return the substring after "non-incremental/"
def trim_instance_path(instance_path):
    # assert "non-incremental/" in instance_path
    assert "non-incremental/" in instance_path
    return instance_path.split("non-incremental/", 1)[1]

def read_csv(csv_path):
    # the csv has a header; read the csv into a dictionary
    csv_dict = {}
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            instance_path = row[1]
            trimmed_instance_path = trim_instance_path(instance_path)
            solved = True if row[2] == "True" else False
            sat = None
            if solved:
                sat = True if row[4] == "sat" else False
            csv_dict[trimmed_instance_path] = (solved, sat)
    return csv_dict
                
def compare_two_csv(target_csv, compare_csv):
    target_dict = read_csv(target_csv)
    compare_dict = read_csv(compare_csv)
    mismatch = False
    for instance_path, (solved, sat) in target_dict.items():
        if solved and (instance_path in compare_dict):
            compare_solved, compare_sat = compare_dict[instance_path]
            if compare_solved and (sat != compare_sat):
                print(f"{instance_path} result mismatch: {sat} in {target_csv} vs {compare_sat} in {compare_csv}")
                mismatch = True
    if not mismatch:
        print(f"Match: {target_csv} and {compare_csv}")

def main():
    for compare_csv in COMPARE_CSV_LST:
        compare_two_csv(TARGET_CSV, compare_csv)

if __name__ == "__main__":
    main()
    
