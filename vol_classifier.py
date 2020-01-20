import csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils import write_raw_score_sk

def read_vol_complete(filename, stage):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames, labels, vols = [], [], []
    if stage == 'train':
        your_list = your_list[:338]
    else:
        your_list = your_list[338:]
    for line in your_list:
        try:
            vol = list(map(float, line[2:]))
        except:
            continue
        filenames.append(line[0])
        label = int(line[1])
        labels.append(label)
        vols.append(vol)
    
    return filenames, labels, vols
   


filenames, y_train, X_train = read_vol_complete('./lookupcsv/ADNI_MRI_VOL.csv', 'train')
filenames, y_test, X_test   = read_vol_complete('./lookupcsv/ADNI_MRI_VOL.csv', 'test')

print(y_test)
print(y_train)

accu_list = []
for i in range(10):
    clf = RandomForestClassifier(max_depth=20)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    accu_list.append(accuracy_score(y_test, y_test_pred))
    f = open('./checkpoint_dir/Vol_RF/raw_score_{}.txt'.format(i), 'w')
    write_raw_score_sk(f, clf.predict_proba(X_test), y_test)
    f.close()
print(float(np.mean(accu_list)), float(np.std(accu_list)))


