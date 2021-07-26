# -*- coding = utf-8 -*-
# @Time : 2021/7/21 12:30
# @Author : 戎昱
# @File : VOC_class.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong

#制作VOC全class的.data文件

import os
files_dir = "D:\VOC2012\VOCdevkit\VOC2012\ImageSets\Main"
files_name = os.listdir(files_dir)

classes_name = set()

for file_name in files_name:
    file_name = file_name.split('_')[0]
    classes_name.add(file_name)

class_file_dir = "D:\VOC2012\VOCdevkit\VOC2012\class.data"
with open(class_file_dir,'w') as f:
    for class_name in classes_name:
        f.write(class_name + '\n')