import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import os, cv2
from tqdm.notebook import tqdm 
import time
from glob import glob


LOAD_TRAIN_DIR = '../Input/data/train/images'
SAVE_TRAIN_DIR = '../Input/data/train/newimages'

device = "cuda"
mtcnn = MTCNN(device = device, thresholds = [0.60, 0.70, 0.70], min_face_size = 80)

def get_box_point(boxes):
    xmin, xmax = int(boxes[0, 0])-15, int(boxes[0, 2])+15
    ymin, ymax = int(boxes[0, 1])-30, int(boxes[0, 3])+30
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > 384: xmax = 384
    if ymax > 512: ymax = 512
    return xmin, xmax, ymin, ymax

for people_path in tqdm(glob(LOAD_TRAIN_DIR + "/*" )):
    images = []
    images_name = []
    
    #이미지 미니 배치 생성
    for i, img_path in enumerate(glob(people_path + "/*")):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        images_name.append(img_path)

    #normal 이미지 인덱스 탐색
    for i, name in enumerate(images_name):  
        if "normal" in name:
            normal_idx = i
            break
            
    #mtcnn에 미니 배치 집어넣어서 boxes 리스트 받기
    boxes_list, _ = mtcnn.detect(images)
    
    
    for i, boxes in enumerate(boxes_list):
        if boxes is not None: 
#             print("detect_crop")
            xmin, xmax, ymin, ymax = get_box_point(boxes)
            img = images[i][ymin:ymax, xmin:xmax,:]
        else:
            # 얼굴 검출하지 못하면 normal 이미지 얼굴 좌표 사용
            if boxes_list[normal_idx] is not None:
                print("normal_crop",img_path)
                xmin, xmax, ymin, ymax = get_box_point(boxes_list[normal_idx])
                img = images[i][ymin:ymax, xmin:xmax,:]
#                 plt.imshow(img)
#                 plt.show()
            # normal 이미지에서 얼굴 검출 하지 못했다면 미리 세팅한 좌표 사용
            else:
                print("manual_crop",img_path)
                img=images[i][100:400, 100:300, :]
#                 plt.imshow(img)
#                 plt.show()
        
        #이미지 저장
        people_folder = os.path.join(SAVE_TRAIN_DIR, people_path.split("/")[-1])
        if not os.path.exists(people_folder):
            os.makedirs(people_folder)
        plt.imsave(os.path.join(people_folder, images_name[i].split("/")[-1]), img)