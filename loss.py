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
        num = prediction.size()[0]
        c_mask = true_labels[:, :, :, 4] != 0
        n_mask = true_labels[:, :, :, 4] == 0
        c_mask = c_mask.unsqueeze(-1).expand_as(true_labels)
        n_mask = n_mask.unsqueeze(-1).expand_as(true_labels)

        box_index = 5 * self.B

        pred_c = prediction[c_mask].view(-1, label_length)
        pred_box = pred_c[:, : box_index].contigous().view(-1, 5)
        pred_classes = pred_c[:, box_index:]

        true_c = true_labels[c_mask].view(-1, label_length)
        true_box = true_c[:, : box_index].contigous().view(-1, 5)
        true_classes = true_c[:, box_index:]

        # Compute the loss for predictions with no objects
        n_pred = prediction[n_mask].view(-1, label_length)
        n_true = true_labels[n_mask].view(-1, label_length)
        initial_mask = np.zeros(n_pred.shape)
        initial_n_conf_mask = torch.from_numpy(initial_mask).type(torch.ByteTensor).cuda()
        for i in range(self.B):
            initial_n_conf_mask[:, 4 + i * 5] = 1
        predicted_confidence = n_pred[initial_n_conf_mask]
        true_confidence = n_true[initial_n_conf_mask]
        no_object_loss = F.mse_loss(predicted_confidence, true_confidence, reduction='sum')



