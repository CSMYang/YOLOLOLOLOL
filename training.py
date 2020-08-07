import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from yolo import *
from loss import *
from DataSet import *
from detection_cuda import *
import os
import numpy as np
import math
from datetime import datetime
import xml.etree.ElementTree as ET
import random
import gc
import warnings


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


def update_learning_rate(optimizer, epoch):
    if epoch == 0:
        lr = 3e-4
    elif epoch == 1:
        lr = 3e-4
    elif epoch == 76:
        lr = 1e-4
    elif epoch == 106:
        lr = 1e-5
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    # reason why its 2 iteration is becasue you can leave it on overnight and it will spend approximately 8 hours
    # first argument takes in the folder for the annotations, 2nd argument takes in destination folder, 3rd argument takes in database name
    source_label("data\source_data\VOC2007\Annotations",
                 "data\processed_data", "VOC2007")
    # first argument takes in the output of source label, 2nd argument takes in destination folder, 3rd argument takes in database name
    shuffle_and_split("data\processed_data\VOC2007_label.txt",
                      "data\processed_data", "VOC2007")
    config_file = "cfg\yolov1.cfg"
    yolo = YoloNet(config_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = yolo.to(device)

    Loss_function = LossGetter()

    # initial hyper parmater
    init_lr = 3e-4
    base_lr = 0.01
    momentum = 0.9
    weight_decay = 5.0e-4
    training_cycle = 135
    batch_size = 10
    S = 7
    C = 20
    B = 3

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, yolo.parameters()), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    Train_data_set = FormatedDataSet(True, "data\source_data\VOC2007\JPEGImages",
                                     "data\processed_data\VOC2007_training_label.txt", S, B, C)
    Train_loader = DataLoader(
        Train_data_set, batch_size=batch_size, shuffle=True, num_workers=0)

    Validation_data_set = FormatedDataSet(True, "data\source_data\VOC2007\JPEGImages",
                                          "data\processed_data\VOC2007_validation_label.txt", S, B, C)
    Validation_loader = DataLoader(
        Validation_data_set, batch_size=batch_size, shuffle=True, num_workers=0)

    class_list = []
    class_file = open("data\processed_data\VOC2007_class_label.txt", "r")
    for x in class_file:
        class_list.append(x.strip())
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=Warning)
    print(class_list)
    best_loss = 0
    best_accuray_percentage = 0
    for iteration in range(training_cycle):
        # trianing

        print("iteration {} started".format(iteration))
        yolo.train()
        total_training_loss = 0
        # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for i, dataset in enumerate(Train_loader):
            update_learning_rate(optimizer, iteration)
            img = dataset[0]
            target = dataset[1]
            img = Variable(img).cuda()
            target = Variable(target).cuda()
            # might want to update learning rate, but for now just leave it to see if it works or not
            prediction = yolo(img).cuda()

            loss = Loss_function(prediction, target).cuda()
            loss_value = loss.item()
            total_training_loss += loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del loss, prediction, loss_value, img, target
            if i % 50 == 0:
                # if device.type == 'cuda':
                #     print(torch.cuda.get_device_name(0))
                #     print('Memory Usage:')
                #     print('Allocated:', round(
                #         torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
                #     print('Cached:   ', round(
                #         torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
                print(total_training_loss/(i+1), i)

        # validation
        total_val_loss = 0
        good_count = 0
        bad_count = 0
        for i, dataset in enumerate(Validation_loader):
            img = dataset[0]
            target = dataset[1]
            yolo.eval()
            img = img.cuda()
            target = target.cuda()
            with torch.no_grad():
                prediction = yolo(img).cuda()

                val_loss = Loss_function(prediction, target).cuda()
            if val_loss.item() < 1.5:
                good_count += 1
            else:
                bad_count += 1
            total_val_loss += val_loss.item()
            del img, target, prediction, val_loss
            if i % 150 == 0:
                print(total_val_loss/(i+1), "HI3")
                print(good_count/(good_count+bad_count))
        total_val_loss /= i
        this_iter_precentage = good_count / (good_count + bad_count)

        # please modified the path if it fails to run for torch.save

        torch.save(yolo.state_dict(),
                   "training_result\\latest_state.pth")
        if best_loss == 0 or best_loss >= total_val_loss:
            best_loss = total_val_loss
            best_accuray_percentage = this_iter_precentage
            best_iteration = iteration
            torch.save(yolo.state_dict(),
                       "training_result\\best_state.pth")
            print("replaced at {}".format(iteration))

        print("iteration {} ended".format(iteration),
              total_val_loss, best_loss, best_accuray_percentage, this_iter_precentage)
    print(best_iteration)
