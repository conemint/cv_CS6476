import cv2
import os
import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import math

import stereo

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

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

    calib_settings = {}
    if pic_name == "map":
        imgs = load_images_from_dir(img_dir, ext = ".pgm")

        calib_settings['f'] = 1
        calib_settings['doffs'] = 230
        calib_settings['baseline'] = 1
    else:
        imgs = load_images_from_dir(img_dir)
        imgs.extend(load_images_from_dir(ground_truth_left))
        imgs.extend(load_images_from_dir(ground_truth_right))

        calib = os.path.join(img_dir,"calib.txt")
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

def helper_generate_img(z):

    print("process to normalize and save depth img")
    # normalize
    avg = np.average(z)
    z[z==0] = avg
    z = np.log(z.astype("float32"))
    # print(z, np.min(z), np.max(z))
    normal_array = ((z-np.min(z))/(np.max(z) - np.min(z)))*255
    z_norm = normal_array.astype("uint8")
    # save with JET colormap
    im_color = cv2.applyColorMap(z_norm, cv2.COLORMAP_JET)
    return im_color
    # cv2.imwrite(os.path.join(OUTPUT_DIR,"disparity_%s.png"%img_name), im_color) 


def run_exp_1(img_name = "Jadeplant"):

    '''
        run simple sum squared difference stereo correspondence algorithm
        output:
            basic_d, z:
            also write image to file.
    '''
    imgs, calib = load_set_of_stereo_images(img_name)
    img_cut = []
    # for idx,img in enumerate(imgs):
    #     if idx < 2:
    #         img_cut.append(np.copy(img[:100,:100,:]))
    #     else:
    #         img_cut.append(np.copy(img[:100,:100]))
    # for img in imgs:
    #     print(img.shape)
    # for img in img_cut:
    #     print(img.shape)
    print("init stereo class.")
    st = stereo.stereo(imgs, calib)
    print("running basic algo..")
    basic_d = st.basic_algo_run(x = 9, rg = 40)
    # print("basic_d",basic_d)
    print("converting to depth..")
    z = st.get_z(basic_d)
    # cv2.imshow('img', basic_d)
    # cv2.waitKey(0)
    # cv2.imshow('img', z)
    # cv2.waitKey(0)
    # z_norm = cv2.normalize(z, None, 0, 255, cv2.NORM_MINMAX)

    im_color = helper_generate_img(z)
    cv2.imwrite(os.path.join(OUTPUT_DIR,"disparity_%s.png"%img_name), im_color) 

    # save np.array
    np.save(os.path.join(OUTPUT_DIR,"dsimple_%s"%(img_name)),
        basic_d)

    return basic_d, z


def run_exp_2(img_name = "Jadeplant", ws = 5, rg = 60, L = 20):

    '''
        run Boykov_swap_algo algorithm
        output:
            basic_d, z:
            also write image to file.
    '''
    print("running experiment 2: Boykov_swap_algo algorithm")
    imgs, calib = load_set_of_stereo_images(img_name)
    img_cut = []
    print("init stereo class.")
    st = stereo.stereo(imgs, calib)
    print("running basic algo..")
    basic_d = st.Boykov_swap_algo(ws = ws, rg = rg, L = L, 
        img_name = img_name, outdir = "./test_output1",)
    print("converting to depth..")
    z = st.get_z(basic_d)

    im_color = helper_generate_img(z)
    cv2.imwrite(os.path.join(OUTPUT_DIR,"disparity_%s.png"%img_name), im_color) 

    return basic_d, z

def compare_matrix(image_name = "Piano",method = "SSD", alg2name = "seg_Piano_4"):
    # load ground truth
    truth_dir = os.path.join(TRUTH_LEFT_DIR,image_name)
    truth_fil = os.path.join(truth_dir, "truthD_%s.npy"%image_name)
    truth = np.load(truth_fil)
    print(truth.shape, truth.sum())
    # load simple result
    simple_fil = os.path.join(OUTPUT_DIR, "dsimple_%s.npy"%image_name)
    simple = np.load(simple_fil)
    print(simple.shape, simple.sum())
    # # load energy result
    # energy_fil = os.path.join("./test_output", "%s.npy"%alg2name)
    # energy = np.load(energy_fil)
    # print(energy.shape, energy.sum())

    # get ssd
    ssd1 = ((((truth - simple)**2).sum())**0.5)/truth.size
    ssd2 = 0
    # ssd2 = ((((truth - energy)**2).sum())**0.5)/truth.size
    print("ssd algo1: ", ssd1, "; ssd algo2: ", ssd2)
    return ssd1, ssd2

if __name__ == "__main__":
    # load_set_of_stereo_images("Piano", debug = True)

    n = len(sys.argv) 
    if n == 3:
        img_name = sys.argv[1]
        opt = int(sys.argv[2]) # exp: 1,2
        ws = 5
        rg = 60
    elif n == 1:
        img_name = "Piano"
        opt = 2
        ws = 5
        rg = 60
    elif n == 5:
        img_name = sys.argv[1]
        opt = int(sys.argv[2]) # exp: 1,2
        ws = int(sys.argv[3])
        rg = int(sys.argv[4])

    else:
        print("Use exactly 0 or 2 or 4 arguments. Total arguments passed:", n - 1) 
        raise ValueError

    if opt == 1:
        bd, bz = run_exp_1(img_name)

    elif opt == 2:
        bd, bz = run_exp_2(img_name,  ws = ws, rg = rg, L = 20)
    elif opt == 3:
        compare_matrix(img_name)

