import cv2
import numpy as np


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
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def get_predictions(self, img):
        """
        This function takes an image and gets the predictions
        """
        blobs = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo.setInput(blobs)
        layers = self.yolo.getLayerNames()
        output_layers = [layers[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]
        predictions = self.yolo.forward(output_layers)

        return predictions

    def draw_boxes(self, frame, predictions, color=None):
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
        for index in indexs:
            index = int(index)
            x, y, w, h = boxs[index]
            name = ids[index]
            conf = confidences[index]
            if color is None:
                c = np.random.choice(range(256), size=3).tolist()
            else:
                c = color
            cv2.rectangle(frame, (x, y), (x + w, y + h), c, 2)
            cv2.putText(frame, name, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)
            cv2.putText(frame, str(np.around(conf, 2)), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, c, 2)

    def detect(self, file, video=False, webcam=False):
        """
        This function takes the name of a file, and goes through the entire pipeline.
        """
        if not video:
            img = cv2.imread(file)
            predictions = self.get_predictions(img)
            self.draw_boxes(img, predictions)
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
                    self.draw_boxes(current_frame, predictions, color=(0, 255, 0))
                    cv2.imshow("", current_frame)
                else:
                    print('End of the video reached!')
                    cv2.waitKey(100)
                    break


if __name__ == '__main__':
    detector = Detector()
    detector.detect('p12.jpg')
    detector.detect(None, True, True)
