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
        all_entry = label_txt.strip().split(" ")
        for entry in all_entry:
            pic_name = entry[0]
            self.image.append(image_set_location + "\\" + pic_name)
            entry = entry[1:]
            number_of_boxes = len(entry)/5
            boxes = []
            labels = []
            image_x, image_y, _ = cv2.imread(self.image).shape
            for box_index in range(number_of_boxes):
                xmin = entry[box_index*5 + 5]
                xmax = entry[box_index*5+1 + 5]
                ymin = entry[box_index*5+2 + 5]
                ymax = entry[box_index*5+3 + 5]
                label = entry[box_index*5+4 + 5]
                w = xmax-xmin
                h = ymax-ymin
                x_center = (w)/2 + xmin
                y_center = (h)/2 + ymin
                w /= image_x
                h /= image_y
                x /= image_x
                y /= image y
                # [[x,y,w,h], [w1,y1,w1,h1]...] for ith index picture where x y w h are in portion of the image
                boxes.append([x_center, y_center, w, h])
                # [l1, l2.......] for ith index picture
                labels.append(label)
            self.image_boxes.append(boxes)
            self.label.appen(labels)

    def __getitem__(self, index):
        path = self.path[index]
        img = cv2.imread(path)
        box_vector = self.boxes[index]
        image_label = self.labels[index]

        tensor_vector = Tensor_Vector(box_vector, image_label)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # do we need to normalize these?

        return img, tensor_vector

    def Tensor_Vector(self, vector, label):
        """ this function takes in a vector in the form of
                    [[x,y,w,h], [x1,y1,w1,h1],......]
            and a label in the form of
                    [[l1],[l2], .......]
            and output a tensor vector of the form [s,s,5xB+C]"""

        result = torch.zeros(self.grid_size, self.grid_size,
                             5*self.number_bounding_box+self.number_class)

        for index, box_entry in enumerate(vector):
            x_center = box_entry[0]
            y_center = box_entry[1]
            w = box_entry[2]
            h = box_entry[3]
            label = vector[index]
            # now we need to determine which box in the grid this center belongs too
            x_center = x_center*418
            y_center = y_center*418
            w_center = w*418
            h_center = h*418
            cell_size = 418/7
            i = math.floor(x_center/cell_size)
            j = math.floor(y_center/cell_size)
            # do we take the abs position or the relative position of x,y wrt to to the cell ij that it is located in
            for x in range(self.B):
                k = x*5
                result[i, j, k:k+5] = i, j, w, h, 1
                result[i, j, 5*self.B*5 + label] = 1

            return result
