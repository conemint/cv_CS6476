import numpy as np
import cv2
import os

def convert_color_to_one(a):
    return a[:,:,0] * 0.12 + a[:,:,1] * 0.58 + a[:,:,2]*0.3

class stereo:
    '''
    Args:
        imgs (List[numpy.array]): list of images in order left, 
            right, ground truth left, ground truth right

    Attributes:
        img0 (numpy.array):
        img1 (numpy.array):
        truth_left (numpy.array):
        truth_right (numpy.array):
    '''
    def __init__(self, imgs, calib):
        self.truth_left = imgs[2]
        self.truth_right = imgs[3]
        self.f = calib["f"]
        self.doffs = calib["doffs"]
        self.baseline = calib["baseline"]

        self.img0 = convert_color_to_one(imgs[0])
        self.img1 = convert_color_to_one(imgs[1])

    def SSD(self, a, b, gray = False):
        """
        sum of squared differences measure (SSD)
        Args:
            a (numpy.array): image cut 1
            b (numpy.array): image cut 2
            gray (bolean): if img is grayscale
        Returns:
            int: sum of squared differences
        """
        if a.shape != b.shape:
            raise ValueError

        # if gray:
        #     return ((a-b)**2).sum()

        # aw = a[:,:,0] * 0.12 + a[:,:,1] * 0.58 + a[:,:,2]*0.3
        # bw = b[:,:,0] * 0.12 + b[:,:,1] * 0.58 + b[:,:,2]*0.3
        return ((a-b)**2).sum()

    def cut_seg(self, i,j,x, m, n, img):
        '''
        Args:
            i,j: pixel to cut from
            x: window-size
            m,n: img shape
        Returns:
            list:
                cut(np.array): cut shape
                ij(tuple): relative location
        '''

        cut = img[max(0,i-x):min(m,i+x+1),
            max(0,j-x):min(n,j+x+1)]
        # if i-x >= 0 and j-x>=0:
        #     ij = (x,x)
        # elif i-x<0:
        ij = (min(x,i), min(x,j))
        return [cut,ij]           

    def basic_algo(self, img0, img1, x = 3, rg = 100):
        '''
        basic stereo algorithm of taking a window around 
        every pixel in one image, and search for the best 
        match along the same scan line in the other image. 
        Do this both left to right and right to left.
        Args:
            img0, img1: calculate from img0 to img1
            x (int): window size used around each pixel
                add size of pixels from center to each dirs
                edge of rectangle = 2x+1
            rg (int): search range left&right of pixel
        Returns:
            correspondence (numpy.array):
                [idx, value]
                value being correspondence x index of pixel
        '''
        m, n = self.img0.shape
        print("shape: ", m, n)
        # corr = [np.zeros(shape), np.zeros(shape)]
        corr = np.zeros((m,n,2))
        for i in range(m):
            if i%10 == 0:
                print("row: ",i)
            for j in range(n):
                # for each pixel (i,j)
                # cut = self.img0[max(0,i-x):min(m,i+x+1),
                #     max(0,j-x):min(n,j+x+1)]
                # cut,ij = self.cut_seg(i,j,x, m, n, img0)

                cut = img0[max(0,i-x):min(m,i+x+1),
                    max(0,j-x):min(n,j+x+1)]

                ij = (min(x,i), min(x,j))
                
                w = min(n,j+x+1) - max(0,j-x)
                cand = np.ones((n,)) * (x**2*255) # max val possible
                for k in range(n - w):
                    if k == j:
                        # cand.append(x**2*255) # max val possible
                        continue
                    comp = img1[max(0,i-x):min(m,i+x+1), 
                        k:k+w]
                    ssd = self.SSD(comp, cut)
                    # cand.append(ssd)
                    cand[k] = ssd
                # cand = np.array(cand)
                corr[ij[0],ij[1],0] = np.argmin(cand)
                corr[ij[0],ij[1],1] = cand[np.argmin(cand)]

        # m, n = self.img0.shape
        idx_row = np.array(range(n)).reshape((1,n))
        idx = np.repeat(idx_row, m, axis=0)
        corr[:,:,0] -= idx
        return corr

    def basic_algo_run(self, x = 3, rg = 100):
        print("run left to right: ")
        d_left_right = self.basic_algo(self.img0, self.img1, x = 3, rg = 100)
        print("run right to left: ")
        d_right_left = self.basic_algo(self.img0, self.img1, x = 3, rg = 100)
        # ans = np.copy(d_left_right)
        # return np.minimum(d_left_right, - d_right_left)
        print("get best of 2:")
        d = np.copy(d_left_right[:,:,0])
        m, n = self.img0.shape
        for i in range(m):
            for j in range(n):
                if d_right_left[i,j,1] < d_left_right[i,j,1]:
                    d[i,j] = d_right_left[i,j,0]
        return d

    def get_z(self, d):
        '''
        z = baseline * f / (d + doffs)
        '''
        z = self.baseline * self.f / (self.doffs + d)
        return z
