import os
import pandas as pd
import shutil

# 定义源文件夹和目标文件夹的路径
source_folder = "./ADNI-FINAL"

# 定义 Excel 文件路径
xlsx_file = "../lookupcsv/ADNI-inf.csv"

# 加载 Excel 文件
df = pd.read_csv(xlsx_file)

# 获取第一列的数据
file_names = df['Filename'].tolist()

e = []
df_copy = df.copy()
print(df_copy)
# 遍历文件名列表
for file_name in os.listdir(source_folder):
  f = file_name.split('.')[0]
  if f in file_names:
    e.append(f)
print(e)
for i in  range(len(file_names)):
    if file_names[i] not in e:
        df_copy = df_copy.drop(df_copy[df_copy['Filename'] == file_names[i]].index)
print(df_copy)
df_copy.to_csv('./ADNI-INF-635.csv', index=False)

    