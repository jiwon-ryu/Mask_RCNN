'''
  학습된 모델로 test하는 코드
  
  1. custom_WEIGHTS_PATH = ROOT_DIR + '/logs/cucumber20200730T0341/mask_rcnn_cucumber_xxxx.h5' 부분 수정 (사용할 모델 경로)
      
      log 폴더 내에 있는 가장 최근에 생성한 모델을 쓰려면 아래처럼 쓰면 됨:
      custom_WEIGHTS_PATH = sorted(glob.glob(ROOT_DIR+"/logs/*/mask_rcnn_*.h5"))[-1]
      
  2. save_path = '/data/JW/Mask_RCNN/result_img/' 부분 수정 (마스킹 결과 이미지 저장 위치)
  3. prcurve_path = '/data/JW/Mask_RCNN/pr_curve.jpg' 부분 수정 (Precision-Recall curve 저장 위치)
  
  2020-08-14 Jiwon Ryu
'''

import os
#import cv2
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skimage
import glob
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import cucumber  

# Root directory of the
#  project
ROOT_DIR = os.getcwd() # ~/RJW/Mask_RCNN

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

#custom_WEIGHTS_PATH = sorted(glob.glob(ROOT_DIR+"/logs/*/mask_rcnn_*.h5"))[-1]
custom_WEIGHTS_PATH = ROOT_DIR + '/mask_rcnn_cucumber_sj.h5'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = cucumber.CucumberConfig()   ## cucumber 바꾸기

custom_DIR = os.path.join(ROOT_DIR, "dataset")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax
  
# Load validation or test dataset
dataset = cucumber.CucumberDataset()   
dataset.load_cucumber(custom_DIR, "test")   
# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# load the last model you trained
# weights_path = model.find_last()[1]

# Load weights
print("Loading weights ", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)

from importlib import reload # was constantly changin the visualization, so I decided to reload it instead of notebook
reload(visualize)


##### 결과이미지 출력 / P-R 곡선 / AP 계산 #####
'''
pred_match_list = []   ## 모든 이미지의 pred_match를 모은 1차원 list
gt_match_list = []   ## '' gt_match를 모은 ''
pred_iou_list = []   ## pred_match_list에 대응하는 iou_singleimg를 모은 list
gt_iou_list = []   ## gt_match_list에 ''

AP_csv = []
name_csv = []
'''
# added on Dec 30, 2020
gt_match_sum=[]
pred_match_sum=[]
pred_scores_sum=[]
# 폴더 내 이미지마다 처리
for image_id in dataset.image_ids:
  image, image_meta, gt_class_id, gt_bbox, gt_mask =\
      modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
      
  info = dataset.image_info[image_id]
  print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                         dataset.image_reference(image_id)))

  # Run object detection
  results = model.detect([image], verbose=1)

  # Display results
  ax = get_ax(1)
  r = results[0]
  
  #save_path = '/data/JW/Mask_RCNN/result_img/'   ## 마스킹 된 결과 이미지 저장 경로 설정
  save_path = '/home/mainuser/mount/RJW/Mask_RCNN/result_img/'
  visualize.display_instances(image, info['id'], save_path, r['rois'], r['masks'], r['class_ids'], 
                              dataset.class_names, r['scores'], ax=ax,
                              title="Predictions")   ## 마스킹 된 결과 이미지 저장
    
  gt_match, pred_match, overlaps, iou_singleimg, pred_scores = utils.compute_matches(
        gt_bbox, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.5)   ## iou_singleimg: 2*2 numpy array (row: pred box, col: gt box)  
  
  # added Dec 30, 2020
  gt_match_sum=np.hstack([gt_match_sum,gt_match])
  pred_match_sum=np.hstack([pred_match_sum,pred_match])
  pred_scores_sum=np.hstack([pred_scores_sum,pred_scores])

  # 이미지 별로 csv에 저장
  '''
  precisions_single = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
  recalls_single = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

  precisions_single = np.concatenate([[0], precisions_single, [0]])
  recalls_single = np.concatenate([[0], recalls_single, [recalls_single[-1]]])  
  
  for i in range(len(precisions_single) - 2, -1, -1):
      precisions_single[i] = np.maximum(precisions_single[i], precisions_single[i + 1])

  indices_single = np.where(recalls_single[:-1] != recalls_single[1:])[0] + 1
  AP_single = np.sum((recalls_single[indices_single] - recalls_single[indices_single - 1]) * precisions_single[indices_single])
  AP_csv.append(AP_single)
  name_csv.append(info['id'])
  
  ### END
  for i in range(len(pred_match)):             
      pred_match_list.append(pred_match[i])
      if pred_match[i] > -1:
        pred_iou_list.append(iou_singleimg[i, int(pred_match[i])])            
      else:
        pred_iou_list.append(iou_singleimg[i, np.argmax(iou_singleimg[i,:])])

  for j in range(len(gt_match)):
      gt_match_list.append(gt_match[j])
      if gt_match[j] > -1:
        gt_iou_list.append(iou_singleimg[int(gt_match[j]), j])
      else:
        gt_iou_list.append(iou_singleimg[np.argmax(iou_singleimg[:,j]),j])
  '''


  plt.close()

# added on Dec 30, 2020
indices_score = np.argsort(pred_scores_sum)[::-1]
pred_match_sum = pred_match_sum[indices_score]
precisions = np.cumsum(pred_match_sum > -1) / (np.arange(len(pred_match_sum))+1)
recalls = np.cumsum(pred_match_sum > -1).astype(np.float32) / len(gt_match_sum)
precisions = np.concatenate([[0], precisions, [0]])
recalls = np.concatenate([[0], recalls, [recalls[-1]]])

'''
## 개별 이미지 결과 저장 
AP_csv = np.reshape(AP_csv, (len(AP_csv), 1))
name_csv = np.reshape(name_csv, (len(name_csv), 1))
image_csv = np.concatenate([name_csv, AP_csv], 1)
np.savetxt('image_ap.csv', image_csv, fmt='%s', delimiter=',')
'''

'''
# list to np array
pred_match_list = np.array(pred_match_list)
gt_match_list = np.array(gt_match_list)

# sort match list by iou
pred_sort = np.array(pred_iou_list).argsort()   ## pred_iou_list 오름차순 인덱스
gt_sort = np.array(gt_iou_list).argsort()   ## gt_iou_list 오름차순 인덱스
pred_match_list = pred_match_list[pred_sort[::-1]]   ## pred_match_list를 pred_iou_list 내림차순에 대응되도록 정렬
gt_match_list = gt_match_list[gt_sort[::-1]]   ## gt_match_list를 gt_iou_list 내림차순에 대응되도록 정렬

# precisions and recalls
precisions = np.cumsum(pred_match_list > -1) / (np.arange(len(pred_match_list))+1)
recalls = np.cumsum(pred_match_list > -1).astype(np.float32) / len(gt_match_list)

print('precision: ', precisions[-1])
print('recall: ', recalls[-1])

precisions = np.concatenate([[0], precisions, [0]])
#recalls = np.concatenate([[0], recalls, [1]])
recalls = np.concatenate([[0], recalls, [recalls[-1]]])
'''

for i in range(len(precisions) - 2, -1, -1):   ## range: len-2부터 0까지 -1씩 감소하는 list
  precisions[i] = np.maximum(precisions[i], precisions[i + 1])   ## precision을 단조감소 함수로 만듦

indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
AP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])   ## pr curve의 아래면적 계산
print('AP: ', AP)   ## AP 출력

# save precision-recall curve
#prcurve_path = '/data/JW/Mask_RCNN/pr_curve.jpg'   ## pr curve 이미지로 저장할 경로
prcurve_path = '/home/mainuser/mount/RJW/Mask_RCNN/pr_curve.jpg'
visualize.plot_precision_recall(AP, precisions, recalls, prcurve_path)   ## pr curve 저장

# save precisions & recalls as csv
precisions = precisions.reshape(np.shape(precisions)[0], 1)
recalls = recalls.reshape(np.shape(recalls)[0], 1)
print(np.shape(precisions), np.shape(recalls))
pr_pair = np.concatenate((recalls, precisions), axis=1)
np.savetxt('p-r pair.csv', pr_pair, fmt='%s', delimiter=',')
