import numpy as np
import torch
import cv2
import sys
import time
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from utils.general import check_img_size, non_max_suppression, draw_bbox
from utils.datasets import LoadStreams, LoadImages
from models.experimental import attempt_load



class yolov5_custom():
    def __init__(self, imgsz=416, conf_thres=0.25, iou_thres=0.25):
        self.model = None
        self.dataset = None
        self.imgsz = imgsz  # inference size (pixels)
        self.stride = 64
        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = iou_thres  # NMS IOU threshold
        self.max_det = 100  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.webcam = False
        self.skipFrame = 1

        self.curr_imgs = []
        self.results = []
        self.times = []
        self.xys = []
        self.frame = []

    def load_model(self, weights='weights/best2.pt', imgsz=64):
        stride = 64  # assign defaults
        model = attempt_load(weights)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.model = model
        self.imgsz = imgsz
        self.stride = stride

    def load_dataset(self, source='imgs'):
        self.webcam = source.isnumeric()
        # Dataloader
        if self.webcam:
            dataset = LoadStreams(
                source, img_size=self.imgsz, stride=self.stride)
        else:
            dataset = LoadImages(
                source, img_size=self.imgsz, stride=self.stride)
        self.dataset = dataset

    def detect(self):
        ## save video
        # if self.dataset.isvideo:
        #     w, h, fps = self.dataset.width, self.dataset.height, int(
        #         round(self.dataset.fps[0]))
        #     print("fps:", fps)
        #     fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        #     out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))
        ## save video

        self.curr_imgs.clear()
        self.results.clear()
        self.times.clear()
        self.xys.clear()
        self.frame.clear()

        for count, (path, img, im0s, vid_cap) in enumerate(self.dataset):
            print()
            if count % self.skipFrame != 0:
                continue
            t1 = time.time()

            img = torch.from_numpy(img)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if len(img.shape) == 3:
                img = img[None]
            # Inference
            pred = self.model(img, augment=self.augment, visualize=False)[0]

            # NMS
            pred = non_max_suppression(
                pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # print("***pred : \n", pred, "\n***", end=" ")

            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  
                    p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(
                    ), self.dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(self.dataset, 'frame', 0)

            results, xys, im = draw_bbox(img, im0, pred)
            
            t2 = time.time()
            if len(results) != 0:
                self.curr_imgs.append(im)
                self.results.append(results)
                self.xys.append(xys)
                self.times.append(t2-t1)
                self.frame.append(count)

            # print(f"({t2-t1:.3f}msec)\n\n")

            ## save video
            # if self.dataset.isvideo:
            #     out.write(im0)
            # if cv2.waitKey(1) == 27:
            #     break
        ## save video
        # if self.dataset.isvideo:
        #     out.release()
        # cv2.destroyAllWindows()
