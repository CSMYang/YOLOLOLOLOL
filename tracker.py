import cv2
import numpy as np
from skimage.metrics import structural_similarity
from scipy.spatial import distance
from collections import OrderedDict


class Tracker:
    """
    This class tracks the objects from videos
    """
    def __init__(self, distance_threshold=30, ssim_gaussian=True, ssim_sigma=1.5, dis_count=50):
        """
        This function initializes a tracker.
        """
        self.dist_thresh = distance_threshold
        self.ssim_gaussian = ssim_gaussian
        self.ssim_sigma = ssim_sigma
        self.registered_ids = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_dis_count = dis_count
        self.id_nums = dict()
        self.colors = OrderedDict()

    def compute_ssim(self, frame_1, frame_2, topleft_and_bottomright):
        """
        This function takes two frames and returns the index at where
        the correlation score is the highest.
        """
        frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
        frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
        x_1, y_1, x_2, y_2 = topleft_and_bottomright
        object_ = frame_1[y_1: y_2, x_1, x_2]
        frame_2 = cv2.resize(frame_2, object_.shape)
        ssim = structural_similarity(object_, frame_2, data_range=frame_2.max() - frame_2.min(),
                                     gaussian_weights=self.ssim_gaussian, sigma=self.ssim_sigma)
        return ssim

    def get_spatial_distances(self, boxes):
        """
        This function takes an array of boxes and returns an array of spatial distances
        """
        prev_box_coords = np.array(list(self.registered_ids.values()))
        # for box in boxes:
        #     coord = box[:2]
        #     dist = np.square(prev_box_coords - coord)
        #     dist = np.sum(dist, axis=1)
        #     dist = np.sqrt(dist)
        #     best_index = np.argmin(dist)
        #     if dist[best_index] < self.dist_thresh:
        #         matched_box = self.boxes[best_index, :]
        #         if tuple(matched_box) not in matches:
        #             matches[tuple(matched_box)] = (box, dist[best_index])
        #         elif matches[tuple(matched_box)][1] > dist[best_index]:
        #             matches[tuple(matched_box)] = (box, dist[best_index])
        coords = boxes[:2]
        dist = distance.cdist(prev_box_coords, coords)
        return dist

    def update(self, boxes, names):
        """
        This function takes a frame and boxes of current objects detected, and track the previous
        boxes and assign them with ids.
        """
        if len(boxes) == 0:
            return None
        if len(self.registered_ids) == 0:
            for i in range(len(boxes)):
                if names[i] in self.id_nums:
                    label = names[i] + ' ' + str(self.id_nums[names[i]])
                    self.id_nums[names[i]] += 1
                else:
                    label = names[i] + ' 0'
                    self.id_nums[names[i]] = 1
                self.registered_ids[label] = boxes[i, :]
                self.disappeared[label] = 0
                self.colors[label] = np.random.choice(range(256), size=3).tolist()
        else:
            dist = self.get_spatial_distances(boxes)
            # rows corresponds to boxes in previous boxes
            # cols corresponds to new boxes
            rows = dist.min(axis=1).argsort()
            cols = dist.argmin(axis=1)[rows]
            used_prev_boxes = set()
            used_new_boxes = set()
            registered_ids = list(self.registered_ids.keys())
            for row, col in zip(rows, cols):
                if row in used_prev_boxes or col in used_new_boxes:
                    continue
                label = registered_ids[row]
                self.registered_ids[label] = boxes[col, :]
                self.disappeared[label] = 0
                used_prev_boxes.add(row)
                used_new_boxes.add(col)
            unused_prev_boxes = set(range(0, dist.shape[0])).difference(used_prev_boxes)
            unused_current_boxes = set(range(0, dist.shape[1])).difference(used_new_boxes)

            if len(self.registered_ids) > len(boxes):
                for index in unused_prev_boxes:
                    prev_box_label = registered_ids[index]
                    self.disappeared[prev_box_label] += 1
                    if self.disappeared[prev_box_label] > self.max_dis_count:
                        del self.registered_ids[prev_box_label]
                        del self.disappeared[prev_box_label]
                        del self.colors[prev_box_label]
            else:
                for index in unused_current_boxes:
                    if names[index] in self.id_nums:
                        label = names[index] + ' ' + str(self.id_nums[names[index]])
                        self.id_nums[names[index]] += 1
                    else:
                        label = names[index] + ' 0'
                        self.id_nums[names[index]] = 1
                    self.registered_ids[label] = boxes[index, :]
                    self.disappeared[label] = 0
                    self.colors[label] = np.random.choice(range(256), size=3).tolist()

    def find_matching_object(self, frame, object_image, label=None):
        """
        This function tries to find the object from a frame of current video.
        """
        if len(self.registered_ids) == 0:
            return [], False
        found = False
        n = len(label)
        ssims = []
        names = []
        for name in self.registered_ids:
            if label is None or name[:n] == label:
                x, y, w, h = self.registered_ids[name]
                x_min, x_max = int(x - w / 2), int(x + w / 2)
                y_min, y_max = int(y - h / 2), int(y + h / 2)
                ssim = self.compute_ssim(frame, object_image, [x_min, y_min, x_max, y_max])
                ssims.append(ssim)
                names.append(name)
        index = np.argmax(ssims)
        if ssims[index] > 0.5:
            found = True
        return names[index], found