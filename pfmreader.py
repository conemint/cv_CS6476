#!/usr/local/bin/python3
import sys
import os
import re
from struct import *
import cv2

import pandas as pd
import numpy as np

def read_pfm(file):
    # Adopted from https://stackoverflow.com/questions/48809433/read-pfm-format-in-python
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
    return img, height, width

if __name__ == "__main__":

    # total arguments 
    n = len(sys.argv) 
    print("Total arguments passed:", n) 
    if n != 2:
        raise ValueError
    img_name = sys.argv[1]

    # I/O directories
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "output_images"

    # IMGS_DIR = os.path.join(INPUT_DIR, 'trainingQ')
    TRUTH_LEFT_DIR = os.path.join(INPUT_DIR, 'groundTruthLeft')
    img_dir = os.path.join(TRUTH_LEFT_DIR, img_name)
    img_file = os.path.join(img_dir,"disp0GT.pfm")
    
    # with open(img_file,"rb") as f:
    #     load_pfm(f)
    IMGS_DIR = os.path.join(INPUT_DIR, 'trainingQ')
    calib_dir = os.path.join(IMGS_DIR,img_name)
    calib = os.path.join(calib_dir,"calib.txt")
    # calib_settings = {}
    df = pd.read_csv(calib,header = None)
    focal_length = float(df.iloc[0,0].split("=")[1].split(" ")[4])
    doffs = float(df.iloc[2,0].split("=")[-1])
    baseline = float(df.iloc[3,0].split("=")[-1])

    depth_img, height, width = read_pfm(img_file)
    depth_img = np.array(depth_img)
    # Convert from the floating-point disparity value d [pixels] in the .pfm file to depth Z [mm]
    depths = baseline * focal_length / (depth_img + doffs)
    depths = np.reshape(depths, (height, width))
    depths = np.fliplr([depths])[0]
    np.save(os.path.join(img_dir,"truthD_%s"%img_name),depths)

    avg = np.average(depths)
    depths[depths==0] = avg
    z = np.log(depths.astype("float32"))
    # print(z, np.min(z), np.max(z))
    normal_array = ((z-np.min(z))/(np.max(z) - np.min(z)))*255
    # print(norm)
    # print(normal_array, np.min(normal_array), np.max(normal_array))
    z_norm = normal_array.astype("uint8")
    # print(z_norm)

    # cv2.imshow('img', z_norm)
    # cv2.waitKey(0)
    # np.save(os.path.join(img_dir,"truthD_%s.txt"%img_name), depths)
    # np.savetxt(os.path.join(img_dir,"truthD_%s_log.txt"%img_name), z_norm, delimiter=',', fmt='%d')
    # np.savetxt(os.path.join(img_dir,"truthD_%s.txt"%img_name), depths, delimiter=',', fmt='%d')
    cv2.imwrite(os.path.join(img_dir,"truthD_%s.png"%img_name), z_norm) 