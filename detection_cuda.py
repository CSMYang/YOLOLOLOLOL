"""
Object Detection using the yolo model we implemented.
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import time
from tracker import Tracker
import cv2
from yolo import YoloNet
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Global variables
CONFID = 0.1
PROB = 0.01
NMS = 0.35
IMG_WIDTH = 448
IMG_HEIGHT = 448
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    result = torch.zeros(1, 3, height, width).cuda()
    # print(transforms.ToTensor()(new_img).shape)
    result[0, :, :, :] = transforms.ToTensor()(new_img)
    # result = result.cuda() if torch.cuda.is_available() else result
    return result
    # mean = [122.67891434, 116.66876762, 104.00698793]
    # mean = np.array(mean, dtype=np.float32)
    # # mean = [122, 116, 104]
    # new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
    # new_img = (new_img - mean) / 255
    # new_img = transforms.ToTensor()(new_img)
    # new_img = new_img[None, :, :, :]
    # new_img = Variable(new_img)
    # return new_img


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

    # labels, confidences, scores, boxes = torch.Tensor(0, device=DEVICE), torch.Tensor(0, device=DEVICE), \
    #                                      torch.Tensor(0, device=DEVICE), torch.Tensor(0, 4, device=DEVICE)
    labels, confidences, scores, boxes = [], [], [], []
    # confidence = yolo_output[:, :, 4].unsqueeze(2)
    # for b in range(1, box_num):
    #     confidence = torch.cat((confidence, yolo_output[:, :, 5 * box_num + 4].unsqueeze(2)), 2)
    for i in range(side):
        for j in range(side):
            # print(j, i, box_num)
            score, label = torch.max(yolo_output[j, i, 5 * box_num:], 0)
            # print(score, label)

            for b in range(box_num):
                confidence = yolo_output[j, i, 5 * b + 4]
                print(confidence)
                if float(score) < prob and confidence != 0:
                    continue

                box = yolo_output[j, i, 5 * b: 5 * b + 4]
                # print(box)
                xy_coord = box[:2] / float(side) + torch.Tensor(
                    [i, j]).cuda() / float(side)
                box_coords = torch.Tensor(4).cuda()
                box_coords[:2] = xy_coord - 0.5 * box[2:]
                box_coords[2:] = xy_coord + 0.5 * box[2:]
                box_coords[box_coords < 0] = 0

                # labels = torch.cat((labels, torch.Tensor([label], device=DEVICE)))
                # # print("labels:{}".format(labels))
                # confidences = torch.cat((confidences, torch.Tensor([confidence], device=DEVICE)))
                # scores = torch.cat((scores, torch.Tensor([score], device=DEVICE)))
                # print(boxes.shape, box_coords.shape)
                # boxes = torch.cat((boxes, box_coords))
                labels.append(label)
                confidences.append(confidence)
                scores.append(score)
                boxes.append(box_coords)
    print("finish for loop (get_prediction)")
    if len(labels) == 0:
        return torch.Tensor(0).cuda(), torch.Tensor(0).cuda(), torch.Tensor(0).cuda(), \
            torch.Tensor(0, 4).cuda()
    labels, confidences, scores, boxes = torch.stack(labels, 0), torch.stack(confidences, 0), \
        torch.stack(scores, 0), torch.stack(boxes, 0)

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

    return torch.Tensor(ids, device=DEVICE)


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
        i = boxes_indices.item() if (boxes_indices.numel() ==
                                     1) else boxes_indices[0].item()
        # print(x1[i].item())
        result.append(i)
        if boxes_indices.numel() == 1:
            break

        # https://pytorch.org/docs/master/generated/torch.clamp.html
        intersect_x1 = x1[boxes_indices[1:]].clamp(min=x1[i].item())
        intersect_y1 = y1[boxes_indices[1:]].clamp(min=y1[i].item())
        intersect_x2 = x2[boxes_indices[1:]].clamp(max=x2[i].item())
        intersect_y2 = y2[boxes_indices[1:]].clamp(max=y2[i].item())
        intersect_w = (intersect_x2 - intersect_x1).clamp(min=0)
        intersect_h = (intersect_y2 - intersect_y1).clamp(min=0)

        intersection = intersect_w * intersect_h
        union = areas[i] + areas[boxes_indices[1:]] - intersection
        iou = intersection / union

        # Remove boxes whose iou is higher than the threshold.
        good_ids = (iou <= nms).nonzero().squeeze()
        if good_ids.numel() == 0:
            break
        boxes_indices = boxes_indices[good_ids + 1]

    # Remove boxes whose confidence is lower than the threshold.
    # good_ids = (confidences[result] >= confidence)

    return labels[good_ids], confidences[good_ids], scores[good_ids], boxes[good_ids]


def non_maximum_supression3(labels, confidences, scores, boxes, confidence=CONFID, nms=NMS):
    """
    Apply the non maximum supression to the yolo_output.
    Removes detections with lower object confidence score than confidence.
    :return: A list of tuples including class label, score, confidence, and two box coordinates.
    """
    # if labels.shape[0] == 0:
    #     return torch.Tensor(0).cuda(), torch.Tensor(0).cuda(), torch.Tensor(0).cuda(), \
    #         torch.Tensor(0, 4).cuda()
    indexs = cv2.dnn.NMSBoxes(
        boxes.tolist(), confidences.tolist(), 0.1, nms)
    print("finish nms")
    if len(indexs) == 0:
        return torch.Tensor(0).cuda(), torch.Tensor(0).cuda(), \
               torch.Tensor(0, 4).cuda()
    # new_labels, new_confidences, new_scores, new_boxes = [], [], [], []
    new_labels, new_scores, new_boxes = [], [], []

    for i in indexs:
        new_labels.append(labels[i])
        # new_confidences.append(confidences[i])
        new_scores.append(scores[i])
        new_boxes.append(boxes[i])
        # print(boxes[i].shape)

    # new_labels, new_confidences, new_scores, new_boxes = torch.stack(new_labels, 0).reshape((len(new_labels))), \
    #     torch.stack(new_confidences, 0).reshape((len(new_confidences))), \
    #     torch.stack(new_scores, 0).reshape((len(new_scores))), \
    #     torch.stack(new_boxes, 0).reshape((len(new_boxes), 4))
    new_labels, new_scores, new_boxes = torch.stack(new_labels, 0).reshape((len(new_labels))), \
                                        torch.stack(new_scores, 0).reshape((len(new_scores))), \
                                        torch.stack(new_boxes, 0).reshape((len(new_boxes), 4))
    # print(new_labels, new_confidences, new_scores, new_boxes)
    return new_labels, new_confidences, new_scores, new_boxes


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
    labels, confidences, scores, boxes = get_prediction_from_yolo(
        yolonet(img_input).squeeze(0), side, box_num)
    # labels, confidences, scores, boxes = get_prediction_from_yolo(predictions, side, box_num)
    # NMS
    labels_nms, probs_nms, boxes_nms = torch.Tensor(0, device=DEVICE), torch.Tensor(0, device=DEVICE), \
        torch.Tensor(0, 4, device=DEVICE)
    for i in range(class_num):
        indices = (labels_nms == i)
        if torch.sum(indices) == 0:
            continue
        sublables, subconf, subscores, subboxes = labels[
            indices], confidences[indices], scores[indices], boxes[indices]
        filter_indices = non_maximum_supression(subscores, subboxes)
        labels_nms = torch.cat((labels_nms, sublables[filter_indices]))
        boxes_nms = torch.cat((boxes_nms, subboxes[filter_indices]))
        prob = subconf[filter_indices] * subscores[filter_indices]
        probs_nms = torch.cat((probs_nms, prob))

    # Reformat the result as a list of tuple.
    detect = []
    for i in range(labels_nms.size(0)):
        label, prob, box = labels_nms[i], probs_nms[i], boxes_nms[i]

        x1, y1, x2, y2 = width * box[0], height * \
            box[1], width * box[2], height * box[3]
        detect.append((label, prob, (x1, y1), (x2, y2)))
    return detect


def detect2(yolonet, img, classes):
    """
    Detect objects from the given img.
    :param yolonet: The trained Yolo model.
    :param img: The image with the shape of (448, 448, 3)
    :param classes: The class list for the object detection
    :param width: The width of the image for yolonet input
    :param height: The height of the image for yolonet input
    :return: A list of tuples including class label, score*confidence, and two box coordinates.
    """
    img_input = preprocess_img(img).cuda()
    height, width, _ = img.shape
    box_num = int(yolonet.detection_param['num'])
    side = int(yolonet.detection_param['side'])
    with torch.no_grad():
        yolo_output = yolonet(img_input)
    yolo_output = yolo_output.cpu().data.squeeze(0).cuda()
    labels, confidences, scores, boxes = get_prediction_from_yolo(
        yolo_output, side, box_num)
    # labels, confidences, scores, boxes = get_prediction_from_yolo(predictions, side, box_num)
    # NMS
    labels_nms, scores_nms, boxes_nms = non_maximum_supression3(
        labels, confidences, scores, boxes)

    # Reformat the result as a list of tuple.
    detect = []
    for i in range(labels_nms.size(0)):
        label, prob, box = classes[labels_nms[i]], scores_nms[i], boxes_nms[i]

        x1, y1, x2, y2 = width * box[0], height * \
            box[1], width * box[2], height * box[3]
        detect.append((label, prob, (x1, y1), (x2, y2)))
    return detect


def draw_boxes(img, boxes, color=(0, 255, 0)):
    """
    Visualize the result of object detection.
    :param img: The image
    :param boxes: The result from detect function
    :return: The image with detected objects.
    """
    img_out = img.copy()
    for b in boxes:
        label, prob, bc1, bc2 = b
        # print("coordinates: {}, {}".format(bc1, bc2))
        cv2.rectangle(img_out, bc1, bc2, color=color, thickness=3)
        name = "{} p:{}".format(label, round(float(prob), 2))
        # size, base = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        # tc2 = (bc1[0] + size[0] + 1, bc1[1] + size[1] + base + 1)
        # cv2.rectangle(img_out, bc1, tc2, color=color)
        cv2.putText(img_out, name, ((bc1[0] + bc2[0]) // 3, (bc1[1] + bc2[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.75,
                    color=(255, 255, 255), thickness=2, lineType=8)
        # cv2.putText(img_out, name, ((bc1[0] + bc1[0]) // 2, (bc1[1] + bc2[1]) // 2),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)

    return img_out


def get_boxes(predictions):
    boxes = []
    names = []
    for box in predictions:
        label, prob, bc1, bc2 = box
        w = bc2[0] - bc1[0]
        h = bc2[1] - bc1[1]
        x, y = int(bc1[0] + w / 2), int(bc1[1] + h / 2)
        boxes.append([x, y, w, h])
        names.append(label)
    boxes = np.array(boxes)
    return boxes, names


def track_everything(video_name=None, max_disappear=10):
    """
    This function tries to track every object appeared in a video file
    """
    if video_name is None:
        video_stream = cv2.VideoCapture(0)
    else:
        video_stream = cv2.VideoCapture(video_name)
    tracker = Tracker(dis_count=max_disappear)
    while cv2.waitKey(1) < 0:
        has_frame, current_frame = video_stream.read()
        if has_frame:
            start_time = time.perf_counter()
            predictions = detect2(yolo, current_frame, classes)
            boxes, names = get_boxes(predictions)
            tracker.update(boxes, names)
            for label in tracker.registered_ids:
                x, y, w, h = tracker.registered_ids[label]
                c = tracker.colors[label]
                cv2.rectangle(current_frame, (int(x - w / 2), int(y - h / 2)),
                              (int(x + w / 2), int(y + h / 2)), c, 2)
                cv2.putText(current_frame, label, (x, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)
            time_spent = time.perf_counter() - start_time
            label = 'Current FPS is: %.2f' % (1 / time_spent)
            cv2.putText(current_frame, label, (0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("", current_frame)
        else:
            print('End of the video reached!')
            cv2.waitKey(100)
            break

def track_specific_image(video_name, object_image, ssim_thresh=0.7, class_name=None):
    """
    This function tracks to find a provided object from the video and track it.
    """
    video_stream = cv2.VideoCapture(video_name)
    tracker = Tracker()
    found = False
    Deleted = False
    object_image = cv2.imread(object_image)
    if class_name is None:
        predictions = detect2(yolo, object_image, classes)
        _, names = get_boxes(predictions)
        if len(names) != 0:
            class_name = names[0]
    while cv2.waitKey(1) < 0:
        has_frame, current_frame = video_stream.read()
        if has_frame:
            start_time = time.perf_counter()
            current_frame = cv2.rotate(
                current_frame, cv2.ROTATE_90_CLOCKWISE)
            predictions = detect2(yolo, current_frame, classes)
            boxes, names = get_boxes(predictions)
            tracker.update(boxes, names)
            if not found:
                class_name, found = tracker.find_matching_object(
                    current_frame, object_image, 1, ssim_thresh, class_name)
            if found and not Deleted:
                box = tracker.registered_ids[class_name]
                del tracker.registered_ids[class_name]
                del tracker.disappeared[class_name]
                del tracker.colors[class_name]
                tracker.registered_ids['Target found!'] = box
                tracker.disappeared['Target found!'] = 0
                tracker.colors['Target found!'] = [0, 0, 255]
                Deleted = True
            for label in tracker.registered_ids:
                x, y, w, h = tracker.registered_ids[label]
                c = tracker.colors[label]
                cv2.rectangle(current_frame, (int(x - w / 2), int(y - h / 2)),
                              (int(x + w / 2), int(y + h / 2)), c, 2)
                cv2.putText(current_frame, label, (x, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)

            time_spent = time.perf_counter() - start_time
            label = 'Current FPS is: %.2f' % (1 / time_spent)
            cv2.putText(current_frame, label, (0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("", current_frame)
        else:
            print('End of the video reached!')
            cv2.waitKey(100)
            break

if __name__ == "__main__":
    # Get classes
    classes = get_classes("./data/processed_data/VOC2007_class_label.txt")
    class_num = len(classes)

    # Detect Object
    img_path = "data\\source_data\\VOC2007\\JPEGImages\\009956.jpg"
    config_path = "./cfg/yolov1.cfg"
    weight_path = "data\\training_result\\8\\2best\\best_state.pth"
    yolo = YoloNet(config_path)
    yolo.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    yolo.to(DEVICE)
    yolo.eval()
    video_stream = cv2.VideoCapture('File name here')
    tracking = False
    track_desired_object = False
    desired_object = cv2.imread('File name here')
    while cv2.waitKey(1) < 0:
        has_frame, current_frame = video_stream.read()
        if has_frame:
            start_time = time.perf_counter()
            predictions = detect2(yolo, current_frame, classes)
            current_frame = draw_boxes(current_frame, predictions)
            time_spent = time.perf_counter() - start_time
            label = 'Current FPS is: %.2f' % (1 / time_spent)
            cv2.putText(current_frame, label, (0, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("", current_frame)
        else:
            print('End of the video reached!')
            cv2.waitKey(100)
            break
    # img = cv2.imread(img_path)
    # # print(img.shape)
    # yolo = YoloNet(config_path)
    # yolo.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    # yolo.to(DEVICE)
    # yolo.eval()
    # # result = detect(yolo, img, class_num)
    # result = detect2(yolo, img, classes)
    # print(result)
    #
    # # result = [("car", 0.1, (120, 120), (190, 190))]
    #
    # # Draw boxes
    # if len(result) > 0:
    #     img_out = draw_boxes(img, result)
    #     img_plt_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    #     plt.imshow(img_plt_out)
    #     plt.show()
    track_everything('File name here')
    track_specific_image('File name', 'Object name')