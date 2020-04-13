import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
import math

import stereo

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "./"

IMGS_DIR = os.path.join(INPUT_DIR, 'trainingQ')
TRUTH_LEFT_DIR = os.path.join(INPUT_DIR, 'groundTruthLeft')
TRUTH_RIGHT_DIR = os.path.join(INPUT_DIR, "groundTruthRight")

def load_images_from_dir(data_dir, ext = ".png"):
    """
        helper function to load all png images from dir
    """
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), cv2.IMREAD_UNCHANGED)) for f in imagesFiles]

    return imgs

def load_set_of_stereo_images(pic_name):
    """
        helper function to load img_left, img_right, 
        ground_truth_left,ground_truth_right of 
        spicified picture
        input: 
            pic_name (str): name of picture (folder name)
        output:
            imgs(list of np.array): left, right, truth_l, truth_r
    """
    img_dir = os.path.join(IMGS_DIR,pic_name)
    ground_truth_left = os.path.join(TRUTH_LEFT_DIR,pic_name)
    ground_truth_right = os.path.join(TRUTH_RIGHT_DIR,pic_name)

    imgs = load_images_from_dir(img_dir)
    imgs.extend(load_images_from_dir(ground_truth_left))
    imgs.extend(load_images_from_dir(ground_truth_right))

    # for img in imgs:
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    return imgs

def run_exp_1():

    '''
        run simple sum squared difference stereo correspondence algorithm
    '''
    pass

if __name__ == "__main__":
    load_set_of_stereo_images("Piano")