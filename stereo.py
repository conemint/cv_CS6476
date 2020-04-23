import numpy as np
import cv2
import os
import networkx as nx

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
        # self.img_ds = []

    def pre_process_SSD(self, img0, img1, k = 100, ws = 3):
        """
        pre process img0 and img1 to a list of images
        to calculate disparity with img1 move left/right 
        each one: (a - b)^2 on each pixel
        Then get sum in window via cv2.filter2D
        Args:
            img0 to img1
            k (int): max move pixels
        Output:
            ssd_ls (list): np.arrays C_SSD(d,i,j)
        """
        print("k = ", k)
        # compute list of images:
        # squared differences with move x
        m, n = img0.shape
        img_ds = []
        for x in range(-k+1,k):
            res = np.ones((m,n))*255**2
            if x < 0:
                # print(img0.shape, img1.shape)
                # print(x,"shapes:", res[:,:n+x].shape, img0[:,:n+x].shape, img1[:,n+x:].shape)
                res[:,:n+x] = (img0[:,:n+x] - img1[:,-x:])**2
            elif x > 0:
                res[:,x:] = (img0[:,x:] - img1[:,:n-x])**2
            else:
                res = (img0 - img1)**2
            img_ds.append(res)


        # get window sum
        ssd_ls = []
        kernel = np.ones((ws,ws),np.float32)/ws**2
        for diff_img in img_ds:
            dst = cv2.filter2D(diff_img,-1,kernel)*ws**2
            ssd_ls.append(dst)
        ssd_ls = np.array(ssd_ls)
        return ssd_ls

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

    def basic_algo(self, img0, img1, x = 3, rg = 50):
        '''
        basic stereo algorithm of taking a window around 
        every pixel in one image, and search for the best 
        match along the same scan line in the other image. 
        Do this both left to right and right to left.
        Args:
            img0, img1: calculate from img0 to img1
            x (int): window size of rectangle used around each pixel
                //add size of pixels from center to each dirs
                //edge of rectangle = 2x+1
            rg (int): search range left&right of pixel
        Returns:
            correspondence (numpy.array):
                [idx, value]
                value being correspondence x index of pixel
        '''

        m, n = self.img0.shape
        ssd_ls = self.pre_process_SSD(img0, img1, k = rg, ws = x)

        # choose best for each pixel
        print(ssd_ls.shape)
        corr = np.zeros((m,n,2))
        for i in range(m):
            for j in range(n):
                idx = np.argmin(ssd_ls[:,i,j])
                corr[i,j,:] = [idx - (rg-1),ssd_ls[idx,i,j]]
        return corr

    def basic_algo_run(self, x = 3, rg = 100):
        print("run left to right: ")
        d_left_right = self.basic_algo(self.img0, self.img1, x = x, rg = rg)
        # print("run right to left: ")
        # d_right_left = self.basic_algo(self.img1, self.img0, x = x, rg = rg)
        # # ans = np.copy(d_left_right)
        # # return np.minimum(d_left_right, - d_right_left)
        # print("get best of 2:")
        # d = np.copy(d_left_right[:,:,0])
        # m, n = self.img0.shape
        # for i in range(m):
        #     for j in range(n):
        #         if d_right_left[i,j,1] < d_left_right[i,j,1]:
        #             d[i,j] = -d_right_left[i,j,0]
        # return d

        return d_left_right[:,:,0]

    def get_z(self, d):
        '''
        z = baseline * f / (d + doffs)
        '''
        z = self.baseline * self.f / (self.doffs + abs(d))
        return z

    def build_graph(self):
        '''
        Build graph for a pair of alpha-beta
        with each edge and it's weight.

        '''
        G = nx.DiGraph()
        pass


    def Boykov_sway_algo(self):
        '''

        '''
        # initial labeling: min-unary-cost

        # update cycle
        success = True
        while success:
            success = False
            # for each pair of labels, iterate
            # build graph
            # find min-cut
            # if E(f)_new < E(f)_old, update f, set success = True
            if Ef_new < Ef_old:
                f = f_new
                success = True
        return f
