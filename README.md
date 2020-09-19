Project: Real-time Object Detection with Tracking Module

Authors: Jiahao Cheng 
         Yi Wai Chow 
         Zhiyuan Yang 

![Track without Searching](everything-min.gif)


For detailed explanation, please see our [report](https://github.com/CSMYang/YOLOLOLOLOL/blob/master/Report.pdf).

Our implemention requires cuda.
Open the project under the main directory YOLOLOLOLOL, or it might ouput pathing error

Language version: Python 3.7+
Required packages: numpy, matplotlib, scipy, sys, cv2 (opencv-python: 4.2.0.34+, opencv-contrib-python: 4.3.0.36+)
                   torch, torchvision, skimage

Required images:
000223.jpg
000413.jpg
009757.jpg
009824.jpg
009903.jpg
009957.jpg
009381.jpg

Required videos:
testing.mp4

Required weight file:

   for object tracker:
   yolov3.weights
   yolov3-tiny.weights

   for own implemented yolo:
   best_state.pth -> for good result
   best_state_8-3  -> for bad result
   due to storage limitation, the weight files are not uploaded to github, please train your own weight file
   
Training required file:
   download at http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ for VOC 2007 dataset and put the all the folder inside VOC2007 folder under data directory

How to run:

    1. Download the yolov3 weight and yolov3 tiny.weight file from 
       https://pjreddie.com/media/files/yolov3.weights
       https://pjreddie.com/media/files/yolov3-tiny.weights
       and save it in the main directory
       these 2 weight are used in yolo_cv.py

    2. To check results using YOLO-v1 model, run detection_cuda.py directly.
       For testing, we set CONFID = 0.1, PROD = 0.1, and NMS = 0.35.
       In order to check different results, choose images from ./example and
       manually set the image path in detection_cuda.py. You could also test more images
       using the PASCAL VOC 2007 image dataset or the dataset using the similar format.
       
    3. To check results of tracking, run yolo_cv.py directly. If you want to try different parameters,
       you can do it by giving different thresholds to track_everything or track_specific_image.
       Note that, the tracker implemented in tracker.py is only the backend of the tracking module.
       It needs manual operations to mount that module on detectors. Currently we have it mounted
       on yolo_cv.py and detection_cuda.py

    4. After downloading the dataset, place it in the main directory 
       and follow the comment in the main section of the training code and should be able to run
       if it fails at torch.save, please modified the path in the argument. 
       or create the folder necessary for it to run.

Reference:

For the loss.py, we get the idea from Motokimura's yolo_v1_pytorch code. You can check it [here](https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py).
For the detect part in detection_cuda.py, we get the idea from Motokimura's yolo_v1_pytorch code. You can check it [here](https://github.com/motokimura/yolo_v1_pytorch/blob/master/detect.py).
For the training process in training.py, we get the idea from Motokimura's yolo_v1_pytorch code. You can check it [here](https://github.com/motokimura/yolo_v1_pytorch/blob/master/train_yolo.py).
For the loading YOLO model code in yolo.py, we get the idea from Chaurasia's yolo_v3 code. You can check it [here](https://github.com/AyushExel/Detectx-Yolo-V3/blob/master/DNModel.py).
