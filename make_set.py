# -*- coding = utf-8 -*-
# @Time : 2021/7/26 21:21
# @Author : 戎昱
# @File : make_set.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import os
import random
import os
import numpy as np
import xml.etree.ElementTree as ET
import torch
from tqdm import tqdm
import time

def getGroundTruth(img_path, annotation_path, ClassNameToClassIndex, max_obj):
    ground_truth = np.zeros(shape=(len(img_path), 5 * max_obj))
    ground_truth_index = 0
    for annotation_file in annotation_path:
        # 解析xml文件--标注文件
        tree = ET.parse(annotation_file)
        annotation_xml = tree.getroot()
        # 计算 目标尺寸 对于 原图尺寸 width的比例
        width = (int)(annotation_xml.find("size").find("width").text)
        # 计算 目标尺寸 对于 原图尺寸 height的比例
        height = (int)(annotation_xml.find("size").find("height").text)
        # 一个注解文件可能有多个object标签，一个object标签内部包含一个bnd标签
        objects_xml = annotation_xml.findall("object")
        obj_num = 0
        for object_xml in objects_xml:
            # 获取目标的名字
            class_name = object_xml.find("name").text
            if class_name not in ClassNameToClassIndex:  # 不属于我们规定的类
                continue
            bnd_xml = object_xml.find("bndbox")
            # 目标尺度放缩
            xmin = (int)((float)(bnd_xml.find("xmin").text))
            ymin = (int)((float)(bnd_xml.find("ymin").text))
            xmax = (int)((float)(bnd_xml.find("xmax").text))
            ymax = (int)((float)(bnd_xml.find("ymax").text))
            # 目标中心点
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            # 真实物体的list
            ClassIndex = ClassNameToClassIndex[class_name]

            # yolo的label格式：class，x_center, y_center, w, h
            ground_box = list([ClassIndex, x_center, y_center, w, h])
            ground_truth[ground_truth_index][obj_num * 5:obj_num * 5 + 5] = ground_box

            obj_num = obj_num + 1

        ground_truth_index = ground_truth_index + 1
    ground_truth = torch.Tensor(ground_truth)
    data = [list([img_path[i], ground_truth[i]]) for i in range(len(img_path))]
    return data,ground_truth_index

path = 'dataset'
# 创建文件夹
if not os.path.exists(path):
    os.mkdir(path)
subpath = path + '\\' + 'train'
if not os.path.exists(subpath):
    os.mkdir(subpath)
subpath = path + '\\' + 'test'
if not os.path.exists(subpath):
    os.mkdir(subpath)
subpath = path + '\\' + 'validation'
if not os.path.exists(subpath):
    os.mkdir(subpath)

imgs_dir = "D:\VOC2012\VOCdevkit\VOC2012\JPEGImages"
annotations_dir = "D:\VOC2012\VOCdevkit\VOC2012\Annotations"
ClassesFile = "D:\VOC2012\VOCdevkit\VOC2012\class.data"

img_names = os.listdir(imgs_dir)
img_names.sort()

img_path = []
for img_name in img_names:
        img_path.append(os.path.join(imgs_dir, img_name))

annotation_names = os.listdir(annotations_dir)
annotation_names.sort()  # 图片和文件排序后可以按照相同索引对应

annotation_path = []
for annotation_name in annotation_names:
    annotation_path.append(os.path.join(annotations_dir, annotation_name))

ClassNameToClassIndex = {}
classIndex = 0
with open(ClassesFile, 'r') as classNameFile:
    for className in classNameFile:
        className = className.replace('\n', '')
        ClassNameToClassIndex[className] = classIndex  # 根据类别名制作索引
        classIndex = classIndex + 1
classNum = classIndex  # 一共的类别个数

data, total = getGroundTruth(img_path, annotation_path, ClassNameToClassIndex, 100)
# with open("all_set.txt", "w") as f1:
#     for index in range(0, total):
#         f1.write(data[index][0] + "\n")

img_name = []
for i in range(0, total):
    img_name.append(data[i][0])
    # print(data[i][0])

sum = len(img_name)
train_set = sorted(random.sample(img_name, int(0.8 * sum)))
validation_set = sorted(random.sample(img_name, int(0.1 * sum)))
test_set = sorted(random.sample(img_name, int(0.1 * sum)))
with open("train_set.txt", "w") as f:
    for name in train_set:
        f.write(name + "\n")

with open("validation_set.txt", "w") as f:
    for name in validation_set:
        f.write(name + "\n")

with open("test_set.txt", "w") as f:
    for name in test_set:
        f.write(name + "\n")

# for index in tqdm(range(total),desc='进行中',ncols=100):
#         boxes = np.array(data[index][1]).reshape(-1, 5)  # 清除冗余的[0.,0.,0.,0.,0.]
#         idx = np.argwhere(np.all(boxes[:, ...] == 0, axis=1))
#         boxes = np.delete(boxes, idx, axis=0)
#         np.savetxt("label" + "\\" + data[index][0].split('\\')[-1] + ".txt", np.c_[boxes],
#                    fmt='%f', delimiter='\t')

# sum = len(img_names)
# print(sum)
# train_set = sorted(random.sample(img_names, int(0.8 * sum)))
# validation_set = sorted(random.sample(img_names, int(0.1 * sum)))
# test_set = sorted(random.sample(img_names, int(0.1 * sum)))
# print(len(train_set))
# print(len(validation_set))
# print(len(test_set))
# # with open("all_set.txt", "w") as f:
# #     for name in img_names:
# #         f.write(name + "\n")
# with open(path + '\\' + 'train' + '\\' + "train_set.txt", "w") as f:
#     for name in train_set:
#         f.write(name + "\n")
# with open(path + '\\' + 'validation' + '\\' + "validation_set.txt", "w") as f:
#     for name in validation_set:
#         f.write(name + "\n")
# with open(path + '\\' + 'test' + '\\' + "test_set.txt", "w") as f:
#     for name in test_set:
#         f.write(name + "\n")
