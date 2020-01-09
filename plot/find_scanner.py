import csv
from collections import defaultdict

scanner_csv = './ADNI_scanner.csv'
ADNI_table = '../lookupcsv/ADNI.csv'

scanner_dict = defaultdict(dict)
with open(scanner_csv, 'r') as f:
    reader = csv.reader(f)
    scanner_list = list(reader)[1:]

for line in scanner_list:
    scanner_dict[line[0]][line[1]] = line[4]

with open(ADNI_table, 'r') as f:
    reader = csv.reader(f)
    ADNI_list = list(reader)

find, not_find = 0, 0
ADNI_list[0][-1] = 'scanner'
for line in ADNI_list[1:]:
    filename = line[0]
    key, seq = filename[5:15], filename.split('_')[-2] 
    if key not in scanner_dict:
        not_find += 1
    else:
        seq_dict = scanner_dict[key]
        for seq_key in seq_dict:
            if seq_key == seq or abs(int(seq[1:]) - int(seq_key[1:])) == 1:
                line[-1] = seq_dict[seq_key]
                break

print(not_find)        

with open('./ADNI_1.5T_screening_scanner.csv', 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerows(ADNI_list)
