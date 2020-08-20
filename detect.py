import os
import platform
import shutil
import time
from pathlib import Path
from utils.datasets import  letterbox
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import glob
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from config import get_config
import numpy as np
class Detector(object):
    def __init__(self):
        opt = get_config()
        self.img_size =opt.img_size
        weights= opt.weights
        self.device = opt.device
        self.model = attempt_load(weights, map_location=self.device)
        self.conf_thres=opt.conf_thres
        self.iou_thres=opt.iou_thres
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names


    def detect(self,im0s):

        img = letterbox(im0s, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img) 
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=False)[0]
        box_detects=[]
        ims=[]
        classes=[]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=False, agnostic=False)
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *x, conf, cls in reversed(det):
                    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
                    ims.append(im0s[c1[1]:c2[1],c1[0]:c2[0]])
                    top=c1[1]
                    left=c1[0]
                    right=c2[0]
                    bottom=c2[1]
                    box_detects.append(np.array([left,top, right,bottom]))
                    classes.append(self.names[int(cls)])
        return box_detects,ims,classes



if __name__ == '__main__':

    detector=Detector()
    for path in glob.glob("test/*.jpg"):

        img=cv2.imread(path)
        
        boxes,ims,classes=detector.detect(img)
        print(len(boxes))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        for box,im,lb in zip(boxes,ims,classes):
            img =cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),3,3)
            cv2.putText(img,lb,(box[0],box[1]),font,2,(255,0,0),1)
        cv2.imshow("image",cv2.resize(img,(500,500)))
        cv2.waitKey(0)