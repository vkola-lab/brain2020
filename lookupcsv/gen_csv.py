import csv

def read_txt(filename):
    content = []
    with open(filename, 'r') as f:
        for line in f:
            content.append(line.strip('\n'))
    return content

path = '../../NM/Demor_info/'
filename = [a[:-4] for a in read_txt(path + 'NACC_Data.txt')]
status = read_txt(path + 'NACC_Label.txt')
age = read_txt(path + 'NACC_Age.txt')
gender = read_txt(path + 'NACC_GENDER.txt')
mmse = read_txt(path + 'NACC_MMSE.txt')
apoe = read_txt(path + 'NACC_Apoe.txt')
check = read_txt(path + 'NACC_Check_MLP.txt')

with open('NACC_MLP.csv', 'w') as csvfile:
    fieldnames = ['filename', 'status', 'age', 'gender', 'mmse', 'apoe']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(filename)):
        if check[i] == 'False': continue
        writer.writerow({'filename': filename[i], 'status': status[i], 'age': age[i], \
                        'gender': gender[i], 'mmse': mmse[i], 'apoe': apoe[i]})
    



