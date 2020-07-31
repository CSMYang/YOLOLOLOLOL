import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from yolo import *
from loss import *
import os
import numpy as np
import math
from datetime import datetime
import xml.etree.ElementTree as ET


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


if __name__ == "__main__":
    source_label("data\source_data\VOC2007\Annotations",
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
    num_epochs = 135
    batch_size = 64
    optimizer = torch.optim.SGD(
        yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)
