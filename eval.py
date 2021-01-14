import os
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
import cucumber_map as cucumber
from mrcnn.config import Config

ROOT_DIR = os.getcwd()

sys.path.append(ROOT_DIR)  # To find local version of the library

#custom_WEIGHTS_PATH = sorted(glob.glob(ROOT_DIR+"/logs/*/mask_rcnn_*.h5"))[-1]
custom_WEIGHTS_PATH = '/home/mainuser/mount/HSJ/2101cucumber/models/resnet50_640x640/cucumber20210107T0548/mask_rcnn_cucumber_0016.h5'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

custom_DIR = '/home/mainuser/mount/HSJ/cucumber_dataset'

class InferenceConfig(Config):


    # Give the configuration a recognizable name
    NAME = "cucumber"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cucumber

    DETECTION_MIN_CONFIDENCE = 0.0

    RPN_ACNHOR_RATIOS = [1, 2, 3]

    BACKBONE="resnet50"

    IMAGE_MAX_DIM = 640

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:1"  # /cpu:0 or /gpu:0

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

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

from importlib import reload # was constantly changin the visualization, so I decided to reload it instead of notebook
reload(visualize)

# added on Dec 30, 2020
gt_match_sum=[]
pred_match_sum=[]
pred_scores_sum=[]
gt_match_sum_big=[]
pred_match_sum_big=[]
pred_scores_sum_big=[]
# 폴더 내 이미지마다 처리
for image_id in dataset.image_ids:
  image, image_meta, gt_class_id, gt_bbox, gt_mask =\
      modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
      
  info = dataset.image_info[image_id]

  # Run object detection
  start=time.time()
  results = model.detect([image], verbose=1)
  print(time.time()-start)
  ax = get_ax(1)
  r = results[0]
  k=0
  q=0
  idx_rois=[]
  idx_bbox=[]
  for i in r['rois']:
      if i[2]-i[0]>65:
          idx_rois.append(k)
      k=k+1
  for j in gt_bbox:
      if j[2]-j[0]>65:
          idx_bbox.append(q)
      q=q+1

  big_rois=r['rois'][idx_rois]
  big_masks=r['masks'][:,:,idx_rois]
  big_scores=r['scores'][idx_rois]
  big_class_ids=r['class_ids'][idx_rois]
  big_gt_bbox=gt_bbox[idx_bbox]
  big_gt_class_id=gt_class_id[idx_bbox]
  big_gt_mask=gt_mask[:,:,idx_bbox]


  #save_path = '/data/JW/Mask_RCNN/result_img/'   ## 마스킹 된 결과 이미지 저장 경로 설정
#   save_path = '/home/mainuser/mount/HSJ/2101cucumber/result_img/'
#   visualize.display_instances(image, info['id'], save_path, big_rois, big_masks, big_class_ids, 
#                               dataset.class_names, big_scores, ax=ax,)   ## 마스킹 된 결과 이미지 저장
# plt.close()


  gt_match_big, pred_match_big, overlaps_big, pred_scores_big = utils.compute_matches(
        big_gt_bbox, big_gt_class_id, big_gt_mask,
        big_rois, big_class_ids, big_scores, big_masks, iou_threshold=0.5) 

  gt_match, pred_match, overlaps, pred_scores = utils.compute_matches(
        gt_bbox, gt_class_id, gt_mask,
        r['rois'], r['class_ids'], r['scores'], r['masks'], iou_threshold=0.5) 
  # added Dec 30, 2020
  gt_match_sum=np.hstack([gt_match_sum,gt_match])
  pred_match_sum=np.hstack([pred_match_sum,pred_match])
  pred_scores_sum=np.hstack([pred_scores_sum,pred_scores])
  gt_match_sum_big=np.hstack([gt_match_sum_big,gt_match_big])
  pred_match_sum_big=np.hstack([pred_match_sum_big,pred_match_big])
  pred_scores_sum_big=np.hstack([pred_scores_sum_big,pred_scores_big])

  plt.close()


def mAP_calculation(pred_scores_sum, pred_match_sum, gt_match_sum, savename):
    indices_score = np.argsort(pred_scores_sum)[::-1]
    pred_match_sum = pred_match_sum[indices_score]
    precisions = np.cumsum(pred_match_sum > -1) / (np.arange(len(pred_match_sum))+1)
    recalls = np.cumsum(pred_match_sum > -1).astype(np.float32) / len(gt_match_sum)
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

# Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                    precisions[indices])

    precisions = precisions.reshape(np.shape(precisions)[0], 1)
    recalls = recalls.reshape(np.shape(recalls)[0], 1)
    pr_pair = np.concatenate((recalls, precisions), axis=1)
    np.savetxt(savename, pr_pair, fmt='%s', delimiter=',')

    return mAP

mAP_big=mAP_calculation(pred_scores_sum_big,pred_match_sum_big, gt_match_sum_big, savename='prcurve_big.csv')
mAP=mAP_calculation(pred_scores_sum, pred_match_sum, gt_match_sum, savename='prcurve.csv')

print(mAP_big)
print(mAP)


