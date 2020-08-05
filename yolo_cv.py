import cv2
import numpy as np
from tracker import Tracker
import time


class Detector:
    """
    This is the class of an object detector
    """

    def __init__(self, labels='coco.names', weights='yolov3.weights', cfg='cfg/yolov3.cfg', cuda=False):
        """
        Initializes an object detector
        """
        self.yolo = cv2.dnn.readNet(weights, cfg)
        names = open(labels, 'r')
        self.classes = []
        for line in names:
            self.classes.append(line.strip())
        names.close()
        if cuda:
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    def get_predictions(self, img):
        """
        This function takes an image and gets the predictions
        """
        blobs = cv2.dnn.blobFromImage(
            img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo.setInput(blobs)
        layers = self.yolo.getLayerNames()
        output_layers = [layers[i[0] - 1]
                         for i in self.yolo.getUnconnectedOutLayers()]
        predictions = self.yolo.forward(output_layers)

        return predictions

    def get_boxes(self, frame, predictions, color=None, need_result=False, draw=True):
        """
        This function takes an image and a set of predictions and draws the corresponding
        bounding boxes.
        """
        height, width, _ = frame.shape
        boxs = []
        confidences = []
        ids = []
        for pred in predictions:
            for item in pred:
                class_index = np.argmax(item[5:])
                class_name = self.classes[class_index]
                confidence = item[5:][class_index]
                if confidence > 0.5:
                    x, y, w, h = item[:4]
                    x, w = int(x * width), int(w * width)
                    y, h = int(y * height), int(h * height)
                    top_left_corner_x = int(x - w / 2)
                    top_left_corner_y = int(y - h / 2)

                    boxs.append([top_left_corner_x, top_left_corner_y, w, h])
                    confidences.append(float(confidence))
                    ids.append(class_name)
        indexs = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
        if need_result:
            best_boxes = []
            best_names = []
        for index in indexs:
            index = int(index)
            x, y, w, h = boxs[index]
            name = ids[index]
            if need_result:
                centroid_x, centroid_y = int(x + w / 2), int(y + h / 2)
                best_boxes.append([centroid_x, centroid_y, w, h])
                best_names.append(name)
            conf = confidences[index]
            if draw:
                if color is None:
                    c = np.random.choice(range(256), size=3).tolist()
                else:
                    c = color
                cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
                cv2.putText(frame, name, (x, y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)
                cv2.putText(frame, str(np.around(conf, 2)),
                            (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)
        if need_result:
            return np.array(best_boxes), best_names

    def detect(self, file, video=False, webcam=False):
        """
        This function takes the name of a file, and goes through the entire pipeline.
        """
        if not video:
            img = cv2.imread(file)
            predictions = self.get_predictions(img)
            self.get_boxes(img, predictions)
            cv2.imshow("", img)
            cv2.waitKey(-1)
        else:
            if webcam:
                print('Webcam enabled. Input video ignored.')
                video_stream = cv2.VideoCapture(0)
            else:
                video_stream = cv2.VideoCapture(file)
            while cv2.waitKey(1) < 0:
                has_frame, current_frame = video_stream.read()
                if has_frame:
                    predictions = self.get_predictions(current_frame)
                    self.get_boxes(current_frame, predictions,
                                   color=(0, 255, 0))
                    t, _ = self.yolo.getPerfProfile()
                    label = 'Current FPS is: %.2f' % (
                        cv2.getTickFrequency() / t)
                    cv2.putText(current_frame, label, (0, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    cv2.imshow("", current_frame)
                else:
                    print('End of the video reached!')
                    cv2.waitKey(100)
                    break

    def track_everything(self, video_name=None, max_disappear=10):
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
                predictions = self.get_predictions(current_frame)
                boxes, names = self.get_boxes(
                    current_frame, predictions, need_result=True, draw=False)
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

    def track_specific_image(self, video_name, object_image, ssim_thresh=0.7, class_name=None):
        """
        This function tracks to find a provided object from the video and track it.
        """
        video_stream = cv2.VideoCapture(video_name)
        tracker = Tracker()
        found = False
        Deleted = False
        object_image = cv2.imread(object_image)
        if class_name is None:
            predictions = self.get_predictions(object_image)
            _, names = self.get_boxes(
                object_image, predictions, need_result=True, draw=False)
            if len(names) != 0:
                class_name = names[0]
        while cv2.waitKey(1) < 0:
            has_frame, current_frame = video_stream.read()
            if has_frame:
                start_time = time.perf_counter()
                current_frame = cv2.rotate(
                    current_frame, cv2.ROTATE_90_CLOCKWISE)
                predictions = self.get_predictions(current_frame)
                boxes, names = self.get_boxes(
                    current_frame, predictions, need_result=True, draw=False)
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


if __name__ == '__main__':
    detector = Detector(cuda=False)
    # detector.detect('Capture.PNG')
    # detector.detect(None, True, True)
    detector.track_specific_image("testing.mp4", "Capture.PNG")
    # detector.track_everything("testing.mp4")
