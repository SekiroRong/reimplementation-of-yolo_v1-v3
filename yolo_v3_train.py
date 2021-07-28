# -*- coding = utf-8 -*-
# @Time : 2021/7/26 16:55
# @Author : 戎昱
# @File : yolo_v3_train.py
# @Software : PyCharm
# @Contact : sekirorong@gmail.com
# @github : https://github.com/SekiroRong
from __future__ import division

import os
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from yolo_v3_model import load_model
from yolo_v3_dataset import _create_data_loader, _create_validation_data_loader
from yolo_v3_loss import compute_loss
from yolo_v3_logger import Logger
from yolo_v3_test import _evaluate

from terminaltables import AsciiTable

def to_cpu(tensor):
    return tensor.detach().cpu()

def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = 'yolov3.cfg'
    pretrained_weights = None
    n_cpu = 2
    multiscale_training = True
    epochs = 2
    checkpoint_interval = 1
    evaluation_interval = 1

    pretrained_weights = ''

    #似乎是training时显示信息量的控制字
    verbose = True

    #evaluation的门槛
    iou_thres = 0.5
    conf_thres = 0.1
    nms_thres = 0.5

    #VOC数据集路径
    train_dir = "train_set.txt"
    val_dir = "validation_set.txt"
    ClassesFile = "D:\VOC2012\VOCdevkit\VOC2012\class.data"

    #读取class_names
    with open(ClassesFile, "r") as fp:
        class_names = fp.read().splitlines()

    logdir = 'log'

    #load model
    model = load_model(model, pretrained_weights)
    print('load model\n')

    #Tensor_board
    logger = Logger(logdir)

    #待研究
    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # Load training dataloader
    dataloader = _create_data_loader(
        mini_batch_size,
        model.hyperparams['height'],
        n_cpu,
        train_dir,
        ClassesFile,
        multiscale_training=True)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        mini_batch_size,
        model.hyperparams['height'],
        n_cpu,
        val_dir,
        ClassesFile)

    # Create optimizer

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])

    #training
    for epoch in range(epochs):

        print("\n---- Training Model ----")

        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            # Run optimizer

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # # Log the learning rate
                # logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # # ############
            # # Log progress
            # # ############
            # if args.verbose:
            #     print(AsciiTable(
            #         [
            #             ["Type", "Value"],
            #             ["IoU loss", float(loss_components[0])],
            #             ["Object loss", float(loss_components[1])],
            #             ["Class loss", float(loss_components[2])],
            #             ["Loss", float(loss_components[3])],
            #             ["Batch loss", to_cpu(loss).item()],
            #         ]).table)

            # # Tensorboard logging
            # tensorboard_log = [
            #     ("train/iou_loss", float(loss_components[0])),
            #     ("train/obj_loss", float(loss_components[1])),
            #     ("train/class_loss", float(loss_components[2])),
            #     ("train/loss", to_cpu(loss).item())]
            # logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # Save progress

        # # Save model to checkpoint file
        # if epoch % checkpoint_interval == 0:
        #     checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch}.pth"
        #     print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
        #     torch.save(model.state_dict(), checkpoint_path)

        # Evaluate

        if epoch % evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=iou_thres,
                conf_thres=conf_thres,
                nms_thres=nms_thres,
                verbose=verbose
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)
if __name__ == "__main__":
    run()