#!/bin/bash

# 创建 result 目录（如果不存在）
mkdir -p result

# 遍历当前目录下的所有目录（AD、MCI、NC、SCD）
for type_dir in *; do
  # 跳过 convert_files.sh 和 result 文件夹
  if [ "$type_dir" = "convert_files.sh" ] || [ "$type_dir" = "result" ]; then
    continue
  fi
  cd $type_dir
  # 遍历 sub
  for sub in *; do
    cd $sub
    # 查看第一个文件名
    first_file=$(ls | head -n 1)
    suffix="${first_file##*.}"
    # 判断文件名后缀
    if [ "$suffix" = "dcm" ]; then
      # 如果文件名后缀为 .dcm，则执行 dcm2nii *.dcm 命令
      dcm2niix *.dcm
      # 将生成的 .nii 文件移动到 result 目录下
      mv *.nii ../../result/
    elif [ "$suffix" = "IMA" ]; then
      # 如果文件名后缀为 .IMA，则执行 dcm2nii *.IMA 命令
      dcm2niix *.IMA
      # 将生成的 .nii 文件移动到 result 目录下
      mv *.nii ../../result/
    fi
    cd ..
  done
  cd ..
done



