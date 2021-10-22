import sys
import time
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, draw_bbox

t0 = time.time()

weights= 'weights/best2.pt' # model.pt path(s) #'best.pt'# 'result21/weights/best.pt'  # model.pt path(s) 'best.pt'
source = 'imgs'  # file/dir/URL/glob, 0 for webcam
imgsz=400 # inference size (pixels)
conf_thres=0.25 # confidence threshold
iou_thres=0.25  # NMS IOU threshold
max_det=100  # maximum detections per image
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False # augmented inference

webcam = source.isnumeric()

# Load model
stride = 64 # assign defaults
model = attempt_load(weights) # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Dataloader
if webcam:
    dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    bs = len(dataset)  # batch_size
else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    bs = 1  # batch_size

# save video
if dataset.isvideo:
    w, h, fps = dataset.width, dataset.height, int(round(dataset.fps[0]))
    print("fps:",fps)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))


for count, (path, img, im0s, vid_cap) in enumerate(dataset):
    print('n')
    img = torch.from_numpy(img)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if len(img.shape) == 3:
        img = img[None]
        
    # Inference
    t1 = time.time()
    pred = model(img, augment=augment, visualize=False)[0]

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    t2 = time.time()
     
    print("***pred : \n", pred, "\n***",end=" ")
    print(f"({t2-t1:.3f}msec)\n\n")
    
    for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
    
    draw_bbox(img, im0, pred)
    ## save video
    if dataset.isvideo: out.write(im0)
    if cv2.waitKey(1)==27:
        break
    
## save video
if dataset.isvideo: out.release()
cv2.destroyAllWindows()

t0_1 = time.time()
print(f'final. ({t0_1 - t0:.3f}s)')


        
