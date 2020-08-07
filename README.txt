Project: Real-time Object Detection with Tracking Module

Authors: Jiahao Cheng (ID: 1003065737),
         Yi Wai Chow (ID: ),
         Zhiyuan Yang (ID: )

Language version: Python 3.7+
Required packages: numpy, matplotlib, scipy, sys, cv2 (opencv-python: 4.2.0.34+, opencv-contrib-python: 4.3.0.36+)
                   torch, torchvision, skimage

Required images:

Required videos:

Required weight file:

How to run:
    1. Download the weight file from xxx and save it in ./data/training_result/.
    2. To check results using YOLO-v1 model, run detection_cuda.py directly.
       For testing, we set CONFID = 0.1, PROD = 0.03, and NMS = 0.35.
       In order to check different results, choose images from ./example and
       manually set the image path in detection_cuda.py. You could also test more images
       using the PASCAL VOC 2007 image dataset or the dataset using the similar format.
    3. To check results of tracking, run tracker.py directly.

Reference:
    1. For the loss.py, we get the idea from Motokimura's yolo_v1_pytorch code. You can check
       https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py.
    2. For the detect part in detection_cuda.py, we get the idea from Motokimura's yolo_v1_pytorch code.
       You can check https://github.com/motokimura/yolo_v1_pytorch/blob/master/detect.py.
    3. For the training process in training.py, we get the idea from Motokimura's yolo_v1_pytorch code.
       You can check https://github.com/motokimura/yolo_v1_pytorch/blob/master/train_yolo.py.
    4. For the loading YOLO model code in yolo.py, we get the idea from Chaurasia's yolo_v3 code.
       You can check https://github.com/AyushExel/Detectx-Yolo-V3/blob/master/DNModel.py.