from tracker import Tracker
from yolo_cv import Detector
from detection_cuda import *
import argparse
import cv2
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help='Path to the input image')
    ap.add_argument('-v', '--video', required=True,
                    help='A boolean that indicates whether the input image is a video file')
    ap.add_argument('-m', '--model', required=True,
                    help='Name of the model to be used')
    ap.add_argument('-t', '--track', help='Set to true if you want to track objects')
    ap.add_argument('-o', '--object', help='The path to the image of the '
                                           'object to be identified and tracked')
    ap.add_argument('-l', '--label', help='Label of the object to be tracked. '
                                          'Provide this if you want better performance')
    args = vars(ap.parse_args())
    v3 = False
    if args['model'] == 'yolov3':
        detector = Detector()
        v3 = True
    elif args['model'] == 'yolov3-tiny':
        detector = Detector(weights='yolov3.weights', cfg='cfg/yolov3-tiny.cfg')
        v3 = True
    else:
        config_path = "./cfg/yolov1.cfg"
        weight_path = "data\\training_result\\8\\2best\\best_state.pth"
        detector = YoloNet(config_path)
        detector.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        detector.to(DEVICE)
        detector.eval()
    if 'video' in args and args['video'].lower() == 'true':
        video = cv2.VideoCapture(args['image'])
        if args['track']
