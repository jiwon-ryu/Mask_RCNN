"""
Mask R-CNN

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cucumber.py train --dataset=/path/to/cucumber/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 cucumber.py train --dataset=/path/to/cucumber/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 cucumber.py train --dataset=/path/to/cucumber/dataset --weights=imagenet

    # Apply color splash to video using the last weights you trained
    python3 cucumber.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = "/home/mainuser/mount/HSJ/Mask_RCNN"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


# 원하는 대로 모델 configuration할 것. 어떤 설정들이 있는지는 config.py 참고
class CucumberConfig(Config):

    # Give the configuration a recognizable name
    NAME = "cucumber"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cucumber

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    DETECTION_MIN_CONFIDENCE = 0.0

    RPN_ANCHOR_RATIOS = [1, 2, 3]

    BACKBONE="resnet50"

    IMAGE_MAX_DIM = 640


############################################################
#  Dataset
############################################################

class CucumberDataset(utils.Dataset):

    def load_cucumber(self, dataset_dir, subset):
        """Load a subset of the Cucumber dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("cucumber", 1, "cucumber")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "json_file.json")))
        annotations = list(annotations.values())  
        annotations = [a for a in annotations if a['regions']]

        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "cucumber",
                image_id=a['filename'], 
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "cucumber":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cucumber":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




############################################################
#  Training
############################################################
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cucumbers.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cucumber/dataset/",
                        help='Directory of the cucumber dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    dataset_train = CucumberDataset()
    dataset_train.load_cucumber(args.dataset, "train")
    dataset_train.prepare()

    dataset_val = CucumberDataset()
    dataset_val.load_cucumber(args.dataset, "val")
    dataset_val.prepare()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    config = CucumberConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                                model_dir=args.logs)

    # 전체 score 구간에 대한 precision과 recall 값을 구하기 위해서 Inference configuration에서 detection_min_confidence는 0으로 놓아야 함
    class _InfConfig(CucumberConfig):
        IMAGES_PER_GPU = 1
        GPU_COUNT = 1
        DETECTION_MIN_CONFIDENCE = 0.0

    infconfig=_InfConfig()
    config.display()
    infconfig.display()
    model_inference = modellib.MaskRCNN(mode="inference", config=infconfig, model_dir=args.logs)
    mean_average_precision_callback = modellib.MeanAveragePrecisionCallback(model, model_inference, dataset_val, 1,
                                                                        verbose=1)


    augmentation = iaa.SomeOf((0,3), [
        iaa.Sharpen((0.0, 1.0)),       
        iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.SomeOf((1,2), [
            iaa.GaussianBlur(sigma=(0.0,2.0)),
            iaa.GammaContrast((0.75,1.75)),
            iaa.WithBrightnessChannels(iaa.Add((-20, 20)))
        ], random_order=True),
        iaa.Fliplr(0.5),
        iaa.CropAndPad(percent=(-0.15, 0.15))
    ], random_order=True)



    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1000,
                augmentation=augmentation,
                layers="all",
                custom_callbacks=[mean_average_precision_callback])


