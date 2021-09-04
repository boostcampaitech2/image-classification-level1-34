###################
# import packages #
###################

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
from tqdm.notebook import tqdm 
import time
from glob import glob




#################
# Set Data Path #
#################

LOAD_EVAL_DIR = '../Input/data/eval/images'
SAVE_EVAL_DIR = '../Input/data/eval/newimages'


############
# Function #
############


def get_box_point(boxes):
    xmin, xmax = int(boxes[0, 0])-15, int(boxes[0, 2])+15
    ymin, ymax = int(boxes[0, 1])-30, int(boxes[0, 3])+30
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > 384: xmax = 384
    if ymax > 512: ymax = 512
    return xmin, xmax, ymin, ymax


#################
# Set Processor #
#################
    
device = "cuda"
mtcnn = MTCNN(device = device, thresholds = [0.50, 0.50, 0.50], min_face_size = 80, post_process=True)

########
# main #
########

# 저장 디렉토리가 없다면 생성
if not os.path.exists(SAVE_EVAL_DIR):
    os.makedirs(SAVE_EVAL_DIR)
    
for img_path in tqdm(glob(LOAD_EVAL_DIR + "/*" )):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes, _ = mtcnn.detect(img)

    if boxes is not None: 
        xmin, xmax, ymin, ymax = get_box_point(boxes)
        img = img[ymin:ymax, xmin:xmax,:]
    else:
        # normal 이미지에서 얼굴 검출 하지 못했다면 세팅한 좌표 사용
        print("manual_crop",img_path)
        img = img[100:400, 100:300, :]

    #이미지 저장    
    plt.imsave(os.path.join(SAVE_EVAL_DIR, img_path.split("/")[-1]), img) 
