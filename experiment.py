import cv2
import os
import numpy as np
import pandas as pd
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

def load_set_of_stereo_images(pic_name, debug = False):
    """
        helper function to load img_left, img_right, 
        ground_truth_left,ground_truth_right of 
        spicified picture
        input: 
            pic_name (str): name of picture (folder name)
        output:
            List containing:
                imgs (list of np.array): left, right, truth_l, truth_r
                calib_settings (dict): camera calib settings
    """
    img_dir = os.path.join(IMGS_DIR,pic_name)
    ground_truth_left = os.path.join(TRUTH_LEFT_DIR,pic_name)
    ground_truth_right = os.path.join(TRUTH_RIGHT_DIR,pic_name)

    imgs = load_images_from_dir(img_dir)
    imgs.extend(load_images_from_dir(ground_truth_left))
    imgs.extend(load_images_from_dir(ground_truth_right))

    calib = os.path.join(img_dir,"calib.txt")
    calib_settings = {}
    df = pd.read_csv(calib,header = None)
    calib_settings['f'] = float(df.iloc[0,0].split("=")[1].split(" ")[4])
    calib_settings['doffs'] = float(df.iloc[2,0].split("=")[-1])
    calib_settings['baseline'] = float(df.iloc[3,0].split("=")[-1])
    
    if debug:
        print(imgs[2])
        print(calib_settings)
    # for img in imgs:
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    return [imgs,calib_settings]

def run_exp_1():

    '''
        run simple sum squared difference stereo correspondence algorithm
    '''
    imgs, calib = load_set_of_stereo_images("Piano")
    img_cut = []
    for idx,img in enumerate(imgs):
        if idx < 2:
            img_cut.append(np.copy(img[100,100,:]))
        else:
            img_cut.append(np.copy(img[100,100]))
    print("init stereo class.")
    st = stereo.stereo(imgs, calib)
    print("running basic algo.")
    basic_d = st.basic_algo_run()

    z = st.get_z(basic_d)
    cv2.imshow('img', basic_d)
    cv2.waitKey(0)
    cv2.imshow('img', z)
    cv2.waitKey(0)
    return basic_d, z

if __name__ == "__main__":
    # load_set_of_stereo_images("Piano", debug = True)

    bd, bz = run_exp_1()
    print(bd)
    print(bz)