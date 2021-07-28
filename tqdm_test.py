# -*- coding = utf-8 -*-
# @Time : 2021/7/22 15:21
# @Author : 戎昱
# @File : tqdm_test.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
# from tqdm import tqdm
# import time
# d = {'loss':0.2,'learn':0.8}
# for i in tqdm(range(50),desc='进行中',ncols=100,postfix=d): #desc设置名称,ncols设置进度条长度.postfix以字典形式传入详细信息
#     time.sleep(0.1)
#     pass
import numpy as np
path = "label"+"\\"+"2007_000027.jpg.txt"
boxes = np.loadtxt(path).reshape(-1, 5)
print(boxes)
