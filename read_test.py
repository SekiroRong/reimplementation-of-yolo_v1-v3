# -*- coding = utf-8 -*-
# @Time : 2021/7/26 21:36
# @Author : 戎昱
# @File : read_test.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
sum = []
with open("test_set.txt", "r") as f:
    for line in f.readlines():
        sum.append(line.strip('\n'))  #去掉列表中每一个元素的换行符
print(sum)