import csv
import sys
sys.path.insert(1, '../')
from utils import read_csv

filenames, labels = read_csv('./ADNI.csv')
imageid = [name.split('_')[-1][1:] for name in filenames]
vol_list = []

id_vol = {}
with open('/home/sq/Desktop/UASPMVBM.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

for line in your_list[1:]:
    id_vol[line[7]] = line[9:-1]

for id in imageid:
    if id in id_vol:
        vol_list.append(id_vol[id])
    else:
        vol_list.append(['']*117)

# print(your_list[0])
# print(your_list[0][7], len(your_list[0][9:-1]))
with open('ADNI_MRI_VOL.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['filename', 'status'] + your_list[0][9:-1])
    for i in range(len(filenames)):
        spamwriter.writerow(filenames[i:i+1]+labels[i:i+1]+vol_list[i])

