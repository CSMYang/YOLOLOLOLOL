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
from torch.utils.data import Dataset


class FormatedDataSet(Dataset):
    def __init__(self, training, image_set_location, processed_label, S, B, C):

        self.training = training
        self.grid_size = S
        self.number_bounding_box = B
        self.number_class = C
        self.image_size = 418
        self.label_file = processed_label
        self.image_set_location = image_set_location

        self.label = []
        self.image = []
        self.image_boxes = []

        label_txt = open(self.label_file, "r")
        for i, entry in enumerate(label_txt):
            entry = entry.strip().split(" ")
            pic_name = entry[0]
            self.image.append(image_set_location + "\\" + pic_name)
            entry = entry[1:]
            number_of_boxes = len(entry)//5
            boxes = []
            labels = []
            image_x, image_y, _ = cv2.imread(self.image[i]).shape
            for box_index in range(number_of_boxes):
                xmin = int(entry[box_index*5])
                xmax = int(entry[box_index*5+1])
                ymin = int(entry[box_index*5+2])
                ymax = int(entry[box_index*5+3])
                label = int(entry[box_index*5+4])
                w = xmax-xmin
                h = ymax-ymin
                x_center = (w)/2 + xmin
                y_center = (h)/2 + ymin
                w /= image_x
                h /= image_y
                x_center /= image_x
                y_center /= image_y
                # [[x,y,w,h], [w1,y1,w1,h1]...] for ith index picture where x y w h are in portion of the image
                boxes.append([x_center, y_center, w, h])
                # [l1, l2.......] for ith index picture
                labels.append(label)
            self.image_boxes.append(boxes)
            self.label.append(labels)
        self.len = i

    def __getitem__(self, index):
        path = self.image[index]
        img = cv2.imread(path)
        box_vector = self.image_boxes[index]
        image_label = self.label[index]

        tensor_vector = self.Tensor_Vector(box_vector, image_label)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # do we need to normalize these?
        img = torch.from_numpy(img)
        return img, tensor_vector

    def __len__(self):
        return self.len

    def Tensor_Vector(self, vector, label):
        """ this function takes in a vector in the form of
                    [[x,y,w,h], [x1,y1,w1,h1],......]
            and a label in the form of
                    [[l1],[l2], .......]
            and output a tensor vector of the form [s,s,5xB+C]"""

        result = torch.zeros(self.grid_size, self.grid_size,
                             5*self.number_bounding_box+self.number_class)
        print(vector, "------------------")
        for index, box_entry in enumerate(vector):
            print(box_entry)
            x_center = box_entry[index][0]
            y_center = box_entry[index][1]
            w = box_entry[index][2]
            h = box_entry[index][3]
            print(type(w), h)
            label = label[index]
            # now we need to determine which box in the grid this center belongs too
            x_center = x_center*418
            y_center = y_center*418
            w_center = w*418
            h_center = h*418
            cell_size = 418/7
            i = math.floor(x_center/cell_size)
            j = math.floor(y_center/cell_size)
            # do we take the abs position or the relative position of x,y wrt to to the cell ij that it is located in
            for x in range(self.number_bounding_box):
                k = x*5
                print(i, j, w, h, 1)
                result[i, j, k:k+5] = i, j, w, h, 1

                result[i, j, 5*self.number_bounding_box*5 + label] = 1

            return result
