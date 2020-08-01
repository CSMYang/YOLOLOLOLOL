import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from yolo import *
from loss import *
from DataSet import *
import os
import numpy as np
import math
from datetime import datetime
import xml.etree.ElementTree as ET
import random


def source_label(folder, saving_folder, name):
    test_label = open(saving_folder + "\\" + name+'_label.txt', 'w')
    class_label = open(saving_folder+"\\"+name+"_class_label.txt", "w")
    class_list = []
    for xml_filename in os.listdir(folder):
        # format xmin xmax ymin ymax label_index
        appending_string = ""
        # parse each of the xml file with ET

        root = ET.parse(folder + "\\" + xml_filename).getroot()
        # get the name
        name = root.find("filename").text
        appending_string += name + " "

        for obj in root.findall("object"):

            class_name = obj.find("name").text
            box = obj.find("bndbox")
            x_min = box.find("xmin").text
            x_max = box.find('xmax').text
            y_min = box.find('ymin').text
            y_max = box.find('ymax').text
            if class_name not in(class_list):
                class_list.append(class_name)
            appending_string += x_min + " "
            appending_string += x_max + " "
            appending_string += y_min + " "
            appending_string += y_max + " "
            appending_string += str(class_list.index(class_name)) + " "
        appending_string += "\n"
        test_label.write(appending_string)
    for x in class_list:
        class_label.write(x+"\n")


def shuffle_and_split(processed_data, saving_folder, name):
    data = open(processed_data, "r")
    pre_shuffled_list = []
    for x in data:
        pre_shuffled_list.append(x)
    length = len(pre_shuffled_list)
    random.shuffle(pre_shuffled_list)
    shuffled = pre_shuffled_list
    training_list = shuffled[:int(length*0.7)]
    validation_list = shuffled[int(length*0.7):]

    training_data = open(saving_folder + "\\" + name +
                         '_training_label.txt', 'w')
    validation_data = open(saving_folder + "\\" +
                           name+'_validation_label.txt', 'w')

    for x in training_list:
        training_data.write(x)
    for x in validation_list:
        validation_data.write(x)


if __name__ == "__main__":
    source_label("data\source_data\VOC2007\Annotations",
                 "data\processed_data", "VOC2007")
    shuffle_and_split("data\processed_data\VOC2007_label.txt",
                      "data\processed_data", "VOC2007")
    config_file = "cfg\yolov1.cfg"
    yolo = YoloNet(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo.to(device)

    Loss_function = LossGetter()

    # initial hyper parmater
    init_lr = 0.001
    base_lr = 0.01
    momentum = 0.9
    weight_decay = 5.0e-4
    training_cycle = 100
    batch_size = 64

    optimizer = torch.optim.SGD(
        yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    # assume the DataSet is properly implemneted
    # prob need to come back and fix it
    Train_data_set = FormatedDataSet(True, @"data\source_data\VOC2007\JPEGImages", @"data\processed_data\VOC2007_training_label.txt", yolo.S, yolo.B, yolo.C)
    Train_loader = DataLoader(
        Train_data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    Validation_data_set = FormatedDataSet(True, @"data\source_data\VOC2007\JPEGImages", @"data\processed_data\VOC2007_validation_label.txt", yolo.S, yolo.B, yolo.C)
    Validation_loader = DataLoader(
        Train_data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    for iteration in range(training_cycle):

        # trianing
        total_training_loss = 0
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for i, dataset in enumerate(Train_loader):
            img = dataset[0]
            target = dataset[1]

            # might want to update learning rate, but for now just leave it to see if it works or not
            optimizer.zero_grad()

            prediction = yolo.forward(img)

            loss = Loss_function.forward(prediction, target)

            optimizer.step()

            total_training_loss += loss

        print(total_training_loss/i)

        # validation
        total_val_loss = 0
        best_loss = 0
        for i, dataset in enumerate(Validation_loader):
            img = dataset[0]
            target = dataset[0]

            yolo.eval()

            prediction = yolo.forward(img)

            val_loss = Loss_function.forward(prediction, target)

            total_val_loss += val_loss

        total_val_loss /= i
        if best_loss == 0 or total_val_loss <= best_loss:
            best_loss = total_val_loss
            torch.save(yolo.state_dict(), @"data\training_result")
