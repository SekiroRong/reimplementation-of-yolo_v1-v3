# -*- coding = utf-8 -*-
# @Time : 2021/7/25 18:01
# @Author : 戎昱
# @File : yolo_v3_dataset.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader
import torch
import random
from PIL import Image
from PIL import ImageFile
import random
from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from yolo_v3_augmentations import AUGMENTATION_TRANSFORMS

#设置随机种子
def worker_seed_set(worker_id):
    # See for details of numpy:
    # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
    # See for details of random:
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader

    # NumPy
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))

    # random
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)

#测试用
mytransform = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # 归一化后.不容易产生梯度爆炸的问题
        ])

#暂时只做了加载VOC的数据集
def _create_data_loader(batch_size, img_size,
                        n_cpu,
                        imgs_dir,
                        annotations_dir,
                        ClassesFile,
                        multiscale_training=False):
    """Creates a DataLoader for training.

    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS,
        imgs_dir=imgs_dir,
        annotations_dir=annotations_dir,
        ClassesFile=ClassesFile)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

class ListDataset(Dataset):
    def __init__(self,imgs_dir,annotations_dir,ClassesFile, img_size=416, multiscale=True, transform=None):
        img_names = os.listdir(imgs_dir)
        img_names.sort()
        self.img_path = []
        for img_name in img_names:
            self.img_path.append(os.path.join(imgs_dir, img_name))
        annotation_names = os.listdir(annotations_dir)
        annotation_names.sort()  # 图片和文件排序后可以按照相同索引对应
        self.annotation_path = []
        for annotation_name in annotation_names:
            self.annotation_path.append(os.path.join(annotations_dir, annotation_name))

        self.ClassNameToClassIndex = {}
        classIndex = 0
        with open(ClassesFile, 'r') as classNameFile:
            for className in classNameFile:
                className = className.replace('\n', '')
                self.ClassNameToClassIndex[className] = classIndex  # 根据类别名制作索引
                classIndex = classIndex + 1
        self.classNum = classIndex  # 一共的类别个数

        self.max_obj = 100
        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform
        self.getGroundTruth()
        self.data = [list([self.img_path[i], self.ground_truth[i]]) for i in range(len(self.img_path))]
        self.__getitem__(0)

    def getGroundTruth(self):
        self.ground_truth = np.zeros(shape=(len(self.img_path), 5 * self.max_obj))
        ground_truth_index = 0
        for annotation_file in self.annotation_path:
            ground_truth = []
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
                if class_name not in self.ClassNameToClassIndex:  # 不属于我们规定的类
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
                ClassIndex = self.ClassNameToClassIndex[class_name]

                #yolo的label格式：class，x_center, y_center, w, h
                ground_box = list([ClassIndex, x_center, y_center, w, h])
                # print(ground_box)

                self.ground_truth[ground_truth_index][obj_num*5:obj_num*5+5] = ground_box
                # print(self.ground_truth[ground_truth_index])

                obj_num = obj_num + 1

            ground_truth_index = ground_truth_index + 1
        self.ground_truth = torch.Tensor(self.ground_truth)
    def __getitem__(self, index):
        # height * width * channel
        img_data = cv2.imread(self.data[index][0])
        img_data = cv2.resize(img_data, (448, 448), interpolation=cv2.INTER_AREA)
        boxes = np.array(self.data[index][1]).reshape(-1, 5)#清除冗余的[0.,0.,0.,0.,0.]
        idx = np.argwhere(np.all(boxes[:, ...] == 0, axis=1))
        boxes = np.delete(boxes, idx, axis=0)
        # print(boxes)
        if self.transform:
            try:
                img, bb_targets = self.transform((img_data, boxes))
                # print(bb_targets)
            except Exception:
                print("Could not apply transform.")
                return

        return self.data[index][0], img_data, bb_targets

    def resize(image, size):
        image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        return image
    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_path)

#test
# _create_data_loader(32,448)