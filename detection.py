"""
Object Detection using the yolo model we implemented.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from yolo import YoloNet


CONFID = 0.1
PROB = 0.1
NMS = 0.5


def get_classes(file_path):
    """
    Read the valid classes for object detection from the file based on the file_path.
    """
    f = open(file_path, "r")
    classes = [n.strip() for n in f.readlines()]
    f.close()
    return classes


def preprocess_img(img, width=448, height=448):
    """
    Reorgainze the image so that it could be passed to YoloNet
    """
    x, y, _ = img.shape
    new_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    result = torch.zeros(1, 3, height, width)
    result[0, :, :, :] = torch.from_numpy(np.flipud(new_img))
    result = result.cuda() if torch.cuda.is_available() else result
    return result


def get_prediction_from_yolo(yolo_output, side, box_num, prob=0.1):
    """
    Get object detection result from the yolo output. Return a list of tuples as
    (class_label, score, confidence, box_coord1, box_coord2).
    :param yolo_output: A tensor with the size (S, S, 5 * B + C)
    :return: A list of tuples including class label, score, confidence, and two box coordinates.
    """

    labels, confidences, scores, boxes = torch.Tensor(0), torch.Tensor(0), torch.Tensor(0), torch.Tensor(0, 4)
    for i in range(side):
        for j in range(side):
            score, label = torch.max(yolo_output[j, i, 5 * box_num:], 0)

            for b in range(box_num):
                confidence = yolo_output[j, i, 5 * b + 4]
                if float(confidence * score) < prob:
                    continue

                box = yolo_output[j, i, 5 * b: 5 * b + 4]
                xy_coord = box[:2] * float(side) + torch.Tensor([i, j]) / float(side)
                box_coords = torch.zeros(4)
                box_coords[:2] = xy_coord - 0.5 * box[2:]
                box_coords[2:] = xy_coord + 0.5 * box[2:]

                labels = torch.cat((labels, label))
                confidences = torch.cat((confidences, confidence))
                scores = torch.cat((scores, score))
                boxes = torch.cat((boxes, box_coords))
    return labels, confidences, scores, boxes


def non_maximum_supression(scores, boxes, confidence=0.5, nms=0.4):
    """
    Apply the non maximum supression to the yolo_output.
    Removes detections with lower object confidence score than confidence.
    :return: A list of tuples including class label, score, confidence, and two box coordinates.
    """
    pass


def detect(yoloNet, img):
    """
    Detect objects from the given img.
    :param yoloNet: The trained Yolo model.
    :param img: The image with the shape of (448, 448, 3)
    :return: A list of tuples including class label, score, confidence, and two box coordinates.
    """
    img_input = preprocess_img(img)
    predictions = get_prediction_from_yolo(yoloNet(img).squeeze(0))
    pass


if __name__ == "__main__":
    classes = get_classes("./voc.names")
    class_num = len(classes)
    # BOX_NUM = int(yoloNet.detection_param['num'])
    # SIDE_NUM = int(yoloNet.detection_param['side'])
