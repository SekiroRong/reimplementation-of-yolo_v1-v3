# -*- coding = utf-8 -*-
# @Time : 2021/7/22 15:30
# @Author : 戎昱
# @File : tensorboxX_test.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i**2, global_step=i)
    writer.add_scalar('exponential', 2**i, global_step=i)
writer = SummaryWriter('runs/another_scalar_example')
for i in range(10):
    writer.add_scalar('quadratic', i**3, global_step=i)
    writer.add_scalar('exponential', 3**i, global_step=i)
