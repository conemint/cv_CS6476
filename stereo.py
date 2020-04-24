import numpy as np
import cv2
import os
import sys
import networkx as nx
from itertools import combinations 
import time

def convert_color_to_one(a):
    return a[:,:,0] * 0.12 + a[:,:,1] * 0.58 + a[:,:,2]*0.3

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
        if len(imgs[0].shape) == 3:
            self.img0 = convert_color_to_one(imgs[0])
            self.img1 = convert_color_to_one(imgs[1])
        else:
            self.img0 = imgs[0]
            self.img1 = imgs[1]
        # self.img_ds = []

    def pre_process_SSD(self, img0, img1, k = 100, step = 1, ws = 3):
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
        for x in range(-k+1,k, step):
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
        # print(ssd_ls.shape)
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


    # def V_pq(self, K, p,q,f, pix_to_label,n):
    #     '''
    #     Given adjacent pixel p,q and label assignment f
    #     calculate V(l_p,l_q) = min(K, |l_p - l_q|)
    #     '''
    #     ip,jp = divmod(p,n)
    #     lp = pix_to_label[ip,jp]

    #     iq,jq = divmod(q,n)
    #     lq = pix_to_label[iq,jq]

    #     return min(K, |lp - lq|)

    # def Dp_fp(self, p, pix_to_label, n, ssd_ls):
    #     '''
    #     Calculate D_p(f_p)
    #     '''
    #     i,j = divmod(p,n)
    #     fp = pix_to_label[i,j]
    #     return ssd_ls[fp,i,j]

    def add_cap_tp(self, G, pix, la, lb, ssd_ls, pix_to_label, 
                dirs, m, n, K, tpa = True):
        '''
        Calculate capacity then 
        Add edge alpha - pix to G with capacity
        or pix - beta to G with capacity
        '''

        i,j = divmod(pix,n)
        Dp_alpha = ssd_ls[la,i,j]
        # neighbors
        V = 0
        for di,dj in dirs:
            if not(0<=i+di<m and 0<=j+dj<n):
                continue
            # iq,jq = divmod(q,n)
            lq = pix_to_label[i+di,j+dj]

            # if q in Np
            if lq == la or lq == lb:
                continue
            V += min(K, abs(la - lq))

        tp = Dp_alpha + V
        if tpa:
            G.add_edge('alpha',pix, capacity = tp)
        else:
            G.add_edge(pix,'beta', capacity = tp)
        return tp

    def build_graph(self, alpha, beta, labels,pix_to_label,K, m, n, ssd_ls):
        '''
        Build graph for a pair of alpha-beta
        with each edge and it's weight.
        Args:
            alpha

        '''
        E = 0
        dirs = ((1,0),(0,1), (-1,0),(0,-1), 
            (1,1),(1,-1),(-1,1),(-1,-1))
        G = nx.DiGraph()
        # G.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])
        pix_ab = labels[alpha].copy()
        pix_ab.update(labels[beta])
        # t-links
        # print("build t-links for nodes: ", len(pix_ab))
        for pix in pix_ab:
            tpa = self.add_cap_tp(G, pix, alpha, beta, ssd_ls, pix_to_label, 
                dirs, m, n, K, tpa = True)
            tpb = self.add_cap_tp(G, pix, beta, alpha, ssd_ls, pix_to_label, 
                dirs, m, n, K, tpa = False)
            if pix in labels[alpha]:
                E += tpb
            else:
                E += tpa
        # n-links
        # print("build n-links: ", m*n)
        seen = set()
        for p in range(m*n):
            if p not in pix_ab or p in seen:
                continue
            seen.add(p)
            # process p neighbors
            ip,jp = divmod(p,n)
            for di,dj in dirs:
                if not (0<=ip+di<m and 0<=jp+dj<n):
                    continue
                q = (ip+di)*n + jp+dj
                if q not in pix_ab or q in seen:
                    continue
                seen.add(q)

            # # if not neighbors

            # iq,jq = divmod(q,n)
            # if abs(ip-iq) + abs(iq-iq) != 1:
            #     continue
                # for each pair
                Vab = min(K, abs(alpha - beta))
                G.add_edge(p,q, capacity = Vab)
                G.add_edge(q,p, capacity = Vab)
                if (p in labels[alpha] and q in labels[beta]) or \
                    (p in labels[beta] and q in labels[alpha]):
                    E += Vab
        # if alpha not in G.nodes:
        #     G.add
        return G, E


    def update_pix_to_label(self, pix_to_label, new_labels, ab_list, n):
        '''
        Args:
            pix_to_label(np.array): to update
            new_labels(list[set]): all labels
            ab_list(list[]): list of labels to update
            n(int): img col - used to convert pix and (i,j)
        Output:
            pix_to_label
        '''
        for alpha in ab_list:
            for pix in new_labels[alpha]:
                i,j = divmod(pix,n)
                pix_to_label[i,j] = alpha
        return pix_to_label

    def Ef_total(self, pix_to_label, ssd_ls, dirs, K):
        m, n = self.img0.shape
        E_data = np.zeros((m,n))
        E_smooth = 0
        for i in range(m):
            for j in range(n):
                lp = pix_to_label[i,j]
                E_data[i,j] = ssd_ls[int(lp),i,j]
                for di,dj in dirs:
                    if not(0<=i+di<m and 0<=j+dj<n):
                        continue
                    lq = pix_to_label[i+di,j+dj]
                    E_smooth += min(K, abs(lp - lq))
                     
        Ef = E_data.sum() + E_smooth/2
        return Ef

    def Boykov_swap_algo(self, ws = 3, rg = 100, L = 20, theta = 1e-3, 
        outdir = "./test_output", img_name = "Piano"):
        '''

        '''
        K = L//5
        m, n = self.img0.shape
        dirs = ((1,0),(0,1), (-1,0),(0,-1), 
            (1,1),(1,-1),(-1,1),(-1,-1))

        scale = (rg*2 - 1)//L
        ssd_ls = self.pre_process_SSD(self.img0, self.img1, 
            k = rg, step = scale, ws = ws)

        # initial labeling: 
        # # random
        # pix_to_label = np.random.choice(L, m*n).reshape(m,n)
        # use basic algo
        basic_d = self.basic_algo_run(x = ws, rg = rg)
        pix_to_label = (basic_d + rg-1)//scale
        print(np.min(pix_to_label), np.max(pix_to_label))
        labels = {i: set() for i in range(L)}
        for i in range(m):
            for j in range(n):
                pix = i*n + j
                lb = pix_to_label[i,j]
                labels[lb].add(pix)

        # Ef_old = sys.maxsize

        # calculate energy of whole image with current assignment
        Ef = self.Ef_total(pix_to_label, ssd_ls, dirs, K)
        # Ef_old = [sys.maxsize for i in combinations(labels,2)] 
        # update cycle
        success = True
        cycle = 0
        while success:
            success = False
            cycle += 1
            print("CYCLE: ", cycle)


            # for each pair of labels, iterate
            # sort labels by len of pix in it (descending)
            sorted_label_list = sorted(labels, key=lambda k: len(labels[k]), reverse=True)

            for idx,(alpha, beta) in enumerate(combinations(sorted_label_list,2)):
                # build graph
                if len(labels[alpha]) == 0 and len(labels[beta]) == 0:
                    continue
                G, Ef_old = self.build_graph(alpha, beta,
                    labels,pix_to_label,K, m, n, ssd_ls)
                # print(G.nodes)
                print(idx, "alpha: ", alpha, "; beta: ", beta)
                print("G: # of nodes", len(list(G.nodes)), "# of edges: ", len(list(G.edges)))
                # find min-cut

                
                start_time = time.time()
                cut_value, partition = nx.minimum_cut(G, 'alpha', 'beta')
                print("--- %s seconds ---" % (time.time() - start_time))
                print("cut_value: ", cut_value)
                # print("partition: ")
                # print(type(partition[0]), len(partition[0]), len(partition[1]))
                # if E(f)_new < E(f)_old, update f, set success = True
                # if cut_value + theta < Ef_old :
                Ef_new = self.Ef_total(pix_to_label, ssd_ls, dirs, K)
                print(Ef_new, theta,  Ef)
                if Ef_new + theta < Ef:
                    # strictly better labeling is found
                    imp_pct = ((Ef_old - cut_value)/Ef_old)
                    print("improve:", "{0:.2%}".format(imp_pct))
                    if imp_pct < 0.05:
                        continue

                    # f = f_new
                    partition[0].remove('alpha')
                    partition[1].remove('beta')
                    print("update: ")
                    print("alpha pix: ", len(labels[alpha]), "to: ", len(partition[0]))
                    print("beta pix : ", len(labels[beta]) , "to: ", len(partition[1]))
                    labels[alpha] = partition[0]
                    labels[beta]  = partition[1]

                    pix_to_label = self.update_pix_to_label(pix_to_label, labels, 
                                        [alpha, beta], n)
                    Ef = Ef_new
                    success = True
                    break
            if cycle%1 == 0:
                # save
                pix_to_label_fix = pix_to_label * scale - (rg - 1)
                z = self.get_z(pix_to_label_fix)

                im_color = helper_generate_img(z)
                timestr = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(os.path.join(outdir,"seg_%s_%d_%s.png"%(img_name, cycle, timestr)),
                    im_color) 
                np.save(os.path.join(outdir,"seg_%s_%d_%s"%(img_name, cycle, timestr)),
                    pix_to_label_fix)

        pix_to_label_fix = pix_to_label * scale - (rg - 1)
        return pix_to_label_fix


