import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class LossGetter(nn.Module):
    def __init__(self, S=7, C=20, B=3, coord_scale=5, noobject_scale=0.5):
        """
        Intializaes a loss getter for computing the loss
        :param S: number of sub-grids on each side
        :param C: number of class labels
        :param B: number of bounding box predicted for each sub-grid
        :param coord_scale: coefficient for computing the bounding box loss
        :param noobject_scale: coefficient for computing the non-objectiveness loss
        """
        super(LossGetter, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale

    def iou_score(self, set_1, set_2):
        """
        This function compute the intersection over union score between two sets of bounding boxes
        :param set_1: A set of N bounding boxes. Each row in this tensor the coordinate
                      of top left corner and bottom right corner.
        :param set_2: A set of M bounding boxes.
        :return: A N * M tensor where each element is an IoU score.
        """
        n = set_1.size(0)
        m = set_2.size(0)

        # Gets the coordinate of the top left corner of the intersection
        top_left = torch.max(set_1[:, :2].unsqueeze(1).expand(n, m, 2),
                             set_2[:, :2].unsqueeze(0).expand(n, m, 2))
        # Gets the coordinate of the bottom right corner of the intersection
        bottom_right = torch.min(set_1[:, 2:].unsqueeze(1).expand(n, m, 2),
                                 set_2[:, 2:].unsqueeze(0).expand(n, m, 2))
        # Gets the area of intersection!
        width = bottom_right[:, :, 0] - top_left[:, :, 0]
        width[width < 0] = 0
        height = bottom_right[:, :, 1] - top_left[:, :, 1]
        height[height < 0] = 0
        intersection = width * height

        # Compute area of the unions!
        set_1_heights = set_1[:, 3] - set_1[:, 1]
        set_1_widths = set_1[:, 2] - set_1[:, 0]
        area1 = set_1_heights * set_1_widths

        set_2_heights = set_2[:, 3] - set_2[:, 1]
        set_2_widths = set_2[:, 2] - set_2[:, 0]
        area2 = set_2_heights * set_2_widths

        area1 = area1.unsqueeze(1).expand_as(intersection)
        area2 = area2.unsqueeze(0).expand_as(intersection)
        union = area1 + area2 - intersection

        return intersection / union

    def forward(self, prediction, true_labels):
        """
        This function takes a tensor of predictions and a tensor of true results
        and computes the loss.
        :param prediction: A set of N predictions. Each prediction is a (S, S, B * 5 + C) tensor
        :param true_labels: A set of N true results. Each true label has the same shape as above
        :return: A scalar of loss.
        """
        label_length = self.C + self.B * 5

        # Initialize corresponding masks for selecting predictions
        # with or without objects
        c_mask = true_labels[:, :, :, 4] > 0
        n_mask = true_labels[:, :, :, 4] == 0
        c_mask = c_mask.unsqueeze(-1).expand_as(true_labels)
        n_mask = n_mask.unsqueeze(-1).expand_as(true_labels)

        box_index = 5 * self.B

        pred_c = prediction[c_mask].view(-1, label_length)
        pred_box = pred_c[:, : box_index].contiguous().view(-1, 5)
        pred_classes = pred_c[:, box_index:]

        true_c = true_labels[c_mask].view(-1, label_length)
        true_box = true_c[:, : box_index].contiguous().view(-1, 5)
        true_classes = true_c[:, box_index:]

        # Compute the loss for probabilities.
        prob_loss = F.mse_loss(pred_classes, true_classes)

        # Compute the loss for predictions with objects
        ious = torch.zeros(true_box.size()).cuda()
        responsibility_mask = torch.zeros(
            true_box.size()).type(torch.ByteTensor).cuda()
        responsibility_mask_no_obj = torch.ones(
            true_box.size()).type(torch.ByteTensor).cuda()
        for i in range(0, true_box.size(0), self.B):
            predicted_boxes = pred_box[i: i + self.B]
            pred_coordinates = torch.zeros((predicted_boxes.size()[0], 4),
                                           requires_grad=True).type(torch.FloatTensor).cuda()
            pred_coordinates[:, :2] = predicted_boxes[:, :2] / \
                float(self.S) - predicted_boxes[:, 2:4] / 2
            pred_coordinates[:, 2:4] = predicted_boxes[:,
                                                       :2] / float(self.S) + predicted_boxes[:, 2:4] / 2
            real_box = true_box[i].view(-1, 5)
            real_coordinate = torch.zeros((real_box.size()[0], 4),
                                          requires_grad=True).type(torch.FloatTensor).cuda()
            real_coordinate[:, :2] = real_box[:, :2] / \
                float(self.S) - real_box[:, 2:4] / 2
            real_coordinate[:, 2:4] = real_box[:, :2] / \
                float(self.S) + real_box[:, 2:4] / 2
            iou = self.iou_score(pred_coordinates, real_coordinate)
            best_score, best_index = iou.max(0)
            best_index = best_index.data.cuda()

            responsibility_mask[i + best_index] = 1
            responsibility_mask_no_obj[i + best_index] = 0
            ious[i + best_index, 4] = best_score.data.cuda()

        pred_resp = pred_box[responsibility_mask].view(-1, 5)
        true_resp = true_box[responsibility_mask].view(-1, 5)
        true_iou = ious[responsibility_mask].view(-1, 5)
        coord_loss = F.mse_loss(
            pred_resp[:, :2], true_resp[:, :2], reduction='sum')
        dimension_loss = F.mse_loss(torch.sqrt(
            pred_resp[:, 2:4]), torch.sqrt(true_resp[:, 2:4]), reduction='sum')
        confidence_loss = F.mse_loss(
            pred_resp[:, 4], true_iou[:, 4], reduction='sum')

        # Compute the loss for predictions with no objects
        n_pred = prediction[n_mask].view(-1, label_length)
        n_true = true_labels[n_mask].view(-1, label_length)
        initial_n_conf_mask = torch.zeros(
            n_pred.size()).type(torch.ByteTensor).cuda()
        for i in range(self.B):
            initial_n_conf_mask[:, 4 + i * 5] = 1
        predicted_confidence = n_pred[initial_n_conf_mask]
        true_confidence = n_true[initial_n_conf_mask]
        no_object_loss = F.mse_loss(
            predicted_confidence, true_confidence, reduction='sum')

        total_loss = self.coord_scale * coord_loss + self.coord_scale * dimension_loss \
            + confidence_loss + self.noobject_scale * no_object_loss \
            + prob_loss
        return total_loss / prediction.size(0)
