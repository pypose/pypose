import cv2
import numpy as np
from .CacherDatasetBase import CacherDatasetBase
from .utils import readPFM

class CacherSceneflowDataset(CacherDatasetBase):

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):

        if datatype == 'img0':
            return trajstr + '/left/' + framestr + '.png'
        if datatype == 'img1':
            return trajstr + '/right/' + framestr + '.png'
        if datatype == 'disp0':
            trajstr = trajstr.replace('frames_cleanpass', 'disparity').replace('frames_finalpass', 'disparity') # hard code
            return trajstr + '/left/' + framestr + '.pfm'

    def load_image(self, fn):
        img = cv2.imread(self.dataroot + '/' + fn)
        assert img is not None, "Error loading image {}".format(self.dataroot + '/' + fn)
        return img

    def load_disparity(self, fn):
        dispImg, scale0 = readPFM(self.dataroot + '/' + fn)
        return dispImg

class CacherKittiDataset(CacherDatasetBase):

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):

        if datatype == 'img0':
            return trajstr + '/colored_0/' + framestr + '_10.png'
        if datatype == 'img1':
            return trajstr + '/colored_1/' + framestr + '_10.png'
        if datatype == 'disp0':
            return trajstr + '/disp_occ/' + framestr + '_10.png'

    def load_image(self, fn):
        # import ipdb;ipdb.set_trace()
        img = cv2.imread(self.dataroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert img is not None, "Error loading image {}".format(fn)
        # the kitti stereo image comes in different sizes
        # clip the data, because the cacher buffer cannot deal with difference sizes
        imgh, imgw, _ = img.shape
        imgx = (imgh-370)//2
        imgy = (imgw - 1224)//2
        img = img[imgx:imgx+370, imgy:imgy+1224,:]
        return img

    def load_disparity(self, fn):
        dispImg = cv2.imread(self.dataroot + '/' + fn)
        # print(dispImg.shape)
        imgh, imgw, _ = dispImg.shape
        imgx = (imgh-370)//2
        imgy = (imgw - 1224)//2
        dispImg = dispImg[imgx:imgx+370, imgy:imgy+1224, 0].astype(np.float32)
        return dispImg
