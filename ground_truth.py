'''
    test 폴더의 ground truth annotation을 jpg 파일로 폴더에 저장하는 코드
    prediction 결과 이미지와의 비교 용도로 쓸 수 있음
    
    save_path = '/data/JW/Mask_RCNN/ground_truth/' 부분 수정 후 사용 (jpg 파일 저장할 폴더)
    
    만약 validation에 대해서 보고 싶다면
    dataset.load_cucumber(CUCUMBER_DIR, 'test') 부분의 'test'를 'val'로 수정하면 됨
    
    SOURCE: https://github.com/matterport/Mask_RCNN/blob/v2.1/samples/cucumber/inspect_cucumber_data.ipynb
    
    2020-08-05 Jiwon Ryu
'''

import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.getcwd()   ## ~/Mask_RCNN

# Import Mask RCNN
sys.path.append(ROOT_DIR)
import mrcnn.utils
import mrcnn.visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import cucumber

config = cucumber.CucumberConfig()
CUCUMBER_DIR = os.path.join(ROOT_DIR, 'dataset')

dataset = cucumber.CucumberDataset()
dataset.load_cucumber(CUCUMBER_DIR, 'test')
dataset.prepare()

# 이미지 저장
for image_id in dataset.image_ids:
    # Load random image and mask.
    image = dataset.load_image(image_id)
    info = dataset.image_info[image_id]
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = mrcnn.utils.extract_bboxes(mask)
    
    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    log("image", image)
    log("mask", mask)
    log("class_ids", class_ids)
    log("bbox", bbox)
    
    # 저장
    save_path = '/data/JW/Mask_RCNN/ground_truth/'   ## annotation 이미지 (ground truth) 저장 경로
    mrcnn.visualize.display_instances(image, info['id'], save_path, bbox, mask, class_ids, dataset.class_names)
    plt.close()

