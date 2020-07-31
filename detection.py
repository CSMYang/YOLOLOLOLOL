"""
Object Detection using the yolo model we implemented.
"""
import torch
import numpy as np
import cv2
from yolo import YoloNet

# Global variables
CONFID = 0.1
PROB = 0.1
NMS = 0.5
IMG_WIDTH = 448
IMG_HEIGHT = 448


def get_classes(file_path):
    """
    Read the valid classes for object detection from the file based on the file_path.
    """
    f = open(file_path, "r")
    classes = [n.strip() for n in f.readlines()]
    f.close()
    return classes


def preprocess_img(img, width=IMG_WIDTH, height=IMG_HEIGHT):
    """
    Reorgainze the image so that it could be passed to YoloNet
    """
    x, y, _ = img.shape
    new_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    result = torch.zeros(1, 3, height, width)
    result[0, :, :, :] = torch.from_numpy(np.flipud(new_img))
    result = result.cuda() if torch.cuda.is_available() else result
    return result


def get_prediction_from_yolo(yolo_output, side, box_num, prob=PROB):
    """
    Get object detection result from the yolo output. Return a list of tuples as
    (class_label, score, confidence, box_coord1, box_coord2).
    :param yolo_output: A tensor with the size (S, S, 5 * B + C)
    :param side: S The number of grid
    :param box_num: B the number of bounding boxes
    :param prob: The probability threshold
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


def non_maximum_supression(scores, boxes, confidence=CONFID, nms=NMS):
    """
    Apply the non maximum supression to the yolo_output.
    Removes detections with lower object confidence score than confidence.
    :return: A list of tuples including class label, score, confidence, and two box coordinates.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    sorted_boxes, indices = torch.sort(scores, 0, descending=True)
    result = []
    while indices.numel() > 0:
        i = indices.item() if (indices.numel() == 1) else indices[0]
        result.append(i)
        if indices.numel() == 1:
            break

        intersect_x1 = x1[indices[1:]].clamp(min=x1[i])
        intersect_y1 = y1[indices[1:]].clamp(min=y1[i])
        intersect_x2 = x2[indices[1:]].clamp(max=x2[i])
        intersect_y2 = y2[indices[1:]].clamp(max=y2[i])
        intersect_w = (intersect_x2 - intersect_x1).clamp(min=0)
        intersect_h = (intersect_y2 - intersect_y1).clamp(min=0)

        intersection = intersect_w * intersect_h
        union = areas[i] + areas[indices[1:]] - intersection
        iou = intersection / union

        # Remove boxes whose IoU is higher than the threshold.
        good_ids = (iou <= nms).nonzero().squeeze()
        if good_ids.numel() == 0:
            break
        indices = indices[good_ids + 1]

    return torch.Tensor(ids)


def non_maximum_supression2(labels, confidences, scores, boxes, confidence=CONFID, nms=NMS):
    """
    Apply the non maximum supression to the yolo_output.
    Removes detections with lower object confidence score than confidence.
    :return: A list of tuples including class label, score, confidence, and two box coordinates.
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    sorted_boxes, boxes_indices = torch.sort(scores, 0, descending=True)
    result = []
    while boxes_indices.numel() > 0:
        i = boxes_indices.item() if (boxes_indices.numel() == 1) else boxes_indices[0]
        result.append(i)
        if boxes_indices.numel() == 1:
            break

        intersect_x1 = x1[boxes_indices[1:]].clamp(min=x1[i])
        intersect_y1 = y1[boxes_indices[1:]].clamp(min=y1[i])
        intersect_x2 = x2[boxes_indices[1:]].clamp(max=x2[i])
        intersect_y2 = y2[boxes_indices[1:]].clamp(max=y2[i])
        intersect_w = (intersect_x2 - intersect_x1).clamp(min=0)
        intersect_h = (intersect_y2 - intersect_y1).clamp(min=0)

        intersection = intersect_w * intersect_h
        union = areas[i] + areas[boxes_indices[1:]] - intersection
        iou = intersection / union

        # Remove boxes whose IoU is higher than the threshold.
        good_ids = (iou <= nms).nonzero().squeeze()
        if good_ids.numel() == 0:
            break
        boxes_indices = boxes_indices[good_ids + 1]

    # Remove boxes whose confidence is lower than the threshold.
    good_ids = (confidences[ids] >= confidence)

    return labels[good_ids], confidences[good_ids], scores[good_ids], boxes[good_ids]


def detect(yolonet, img, class_num, width=IMG_WIDTH, height=IMG_HEIGHT):
    """
    Detect objects from the given img.
    :param yolonet: The trained Yolo model.
    :param img: The image with the shape of (448, 448, 3)
    :param class_num: C The number of classes
    :param width: The width of the image for yolonet input
    :param height: The height of the image for yolonet input
    :return: A list of tuples including class label, score*confidence, and two box coordinates.
    """
    img_input = preprocess_img(img)
    box_num = int(yolonet.detection_param['num'])
    side = int(yolonet.detection_param['side'])
    predictions = get_prediction_from_yolo(yolonet(img_input).squeeze(0), side, box_num)
    labels, confidences, scores, boxes = get_prediction_from_yolo(predictions, side, box_num)
    # NMS
    labels_nms, probs_nms, boxes_nms = torch.Tensor(0), torch.Tensor(0), torch.Tensor(0, 4)
    for i in range(class_num):
        indices = (labels_nms == i)
        if torch.sum(indices) == 0:
            continue
        sublables, subconf, subscores, subboxes = labels[indices], confidences[indices], scores[indices], boxes[indices]
        filter_indices = non_maximum_supression(subscores, subboxes)
        labels_nms = torch.cat((labels_nms, sublables[filter_indices]))
        boxes_nms = torch.cat((boxes_nms, subboxes[filter_indices]))
        prob = subconf[filter_indices] * subscores[filter_indices]
        probs_nms = torch.cat((probs_nms, prob))

    # Reformat the result as a list of tuple.
    detect = []
    for i in range(labels_nms.size(0)):
        label, prob, box = labels_nms[i], probs_nms[i], boxes_nms[i]

        x1, y1, x2, y2 = width * box[0], height * box[1], width * box[2], height * box[3]
        detect.append((label, prob, (x1, y1), (x2, y2)))
    return detect


def detect2(yolonet, img, classes, width=IMG_WIDTH, height=IMG_HEIGHT):
    """
    Detect objects from the given img.
    :param yolonet: The trained Yolo model.
    :param img: The image with the shape of (448, 448, 3)
    :param classes: The class list for the object detection
    :param width: The width of the image for yolonet input
    :param height: The height of the image for yolonet input
    :return: A list of tuples including class label, score*confidence, and two box coordinates.
    """
    img_input = preprocess_img(img)
    box_num = int(yolonet.detection_param['num'])
    side = int(yolonet.detection_param['side'])
    predictions = get_prediction_from_yolo(yolonet(img_input).squeeze(0), side, box_num)
    labels, confidences, scores, boxes = get_prediction_from_yolo(predictions, side, box_num)
    # NMS
    labels_nms, confidences_nms, scores_nms, boxes_nms = non_maximum_supression2(labels, confidences, scores, boxes)

    # Reformat the result as a list of tuple.
    detect = []
    for i in range(labels_nms.size(0)):
        label, prob, box = classes[labels_nms[i]], confidences_nms[i] * scores_nms[i], boxes_nms[i]

        x1, y1, x2, y2 = width * box[0], height * box[1], width * box[2], height * box[3]
        detect.append((label, prob, (x1, y1), (x2, y2)))
    return detect


if __name__ == "__main__":
    classes = get_classes("./voc.names")
    class_num = len(classes)
