import cv2
from .CacherDatasetBase import CacherDatasetBase
from .utils import flow16to32, depth_rgba_float32
import numpy as np

class CacherTartanAirDataset(CacherDatasetBase):
    # def __init__(self, datatypes, trajlist, trajlenlist, framelist, datarootdir=""):

    #     super(CacherDatasetBase, self).__init__(datatypes, trajlist, trajlenlist, framelist, \
    #                                             datarootdir = datarootdir)

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):
        '''
        return the file path name wrt the data type and framestr
        '''

        if datatype == 'img0':
            return trajstr + '/image_left/' + framestr + '_left.png'
        if datatype == 'img0blur':
            if framestr=='000000': # we don't have blur for the first image, because we don't have the flow
                return trajstr + '/image_left/000000_left.png'
            else:
                return trajstr + '/image_left_blur_0.5/' + framestr + '_left.png'
        if datatype == 'img1':
            return trajstr + '/image_right/' + framestr + '_right.png'
        if datatype == 'depth0' or datatype == 'disp0':
            return trajstr + '/depth_left/' + framestr + '_left_depth.png'
        if datatype == 'depth1' or datatype == 'disp1':
            return trajstr + '/depth_right/' + framestr + '_right_depth.png'

        if datatype.startswith('flow') or datatype.startswith('fmask'):
            datatype = datatype.replace('fmask', 'flow')
            flownum = 1 if datatype=='flow' else int(datatype[4:])
            if ind_inv <= flownum: # this frame is at the end of the trajectory, flow doesn't exist
                return None
            framestr2 = str(int(framestr) + flownum).zfill(len(framestr))
            return trajstr + '/' + datatype + '/' + framestr + '_' + framestr2 + '_flow.png'

    def load_image(self, fn):
        # print(self.dataroot + '/' + fn) # for debugging
        # img = np.zeros((480,640,3), dtype=np.uint8)
        img = cv2.imread(self.dataroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert img is not None, "Error loading image {}".format(self.dataroot + '/' + fn)
        return img

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        # fn = '' if fn is None else fn
        # print(self.dataroot + '/' + fn) # for debugging
        # flow32 = np.zeros((480, 640, 2), dtype=np.float32)
        # mask = np.zeros((480, 640), dtype=np.uint8)
        if fn is None: 
            return np.zeros((10,10,2),dtype=np.float32), np.zeros((10,10),dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow16 = cv2.imread(self.dataroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert flow16 is not None, "Error loading flow {}".format(self.dataroot + '/' + fn)
        flow32, mask = flow16to32(flow16)
        return flow32, mask

    def load_depth(self, fn):
        # print(self.dataroot + '/' + fn) # for debugging
        # depth = np.zeros((480, 640), dtype=np.float32)
        # print(self.dataroot + '/' + fn) # for debugging
        depth = np.zeros((480, 640), dtype=np.float32)
        depth_rgba = cv2.imread(self.dataroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert depth_rgba is not None, "Error loading depth {}".format(self.dataroot + '/' + fn)
        depth = depth_rgba_float32(depth_rgba)

        return depth

    def load_disparity(self, fn):
        depth = self.load_depth(fn)
        disp = 80.0/depth # hard coded
        return disp

class CacherTartanAirDatasetNoCompress(CacherDatasetBase):

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):
        '''
        return the file path name wrt the data type and framestr
        '''

        if datatype == 'img0':
            return trajstr + '/image_left/' + framestr + '_left.png'
        if datatype == 'img0blur':
            if framestr=='000000': # we don't have blur for the first image, because we don't have the flow
                return trajstr + '/image_left/000000_left.png'
            else:
                return trajstr + '/image_left_blur_0.5/' + framestr + '_left.png'
        if datatype == 'img1':
            return trajstr + '/image_right/' + framestr + '_right.png'
        if datatype == 'depth0' or datatype == 'disp0':
            return trajstr + '/depth_left/' + framestr + '_left_depth.npy'
        if datatype == 'depth1' or datatype == 'disp1':
            return trajstr + '/depth_right/' + framestr + '_right_depth.npy'

        if datatype.startswith('flow') or datatype.startswith('fmask'):
            datatype = datatype.replace('fmask','flow')
            flownum = 1 if datatype=='flow' else int(datatype[4:])
            if ind_inv <= flownum: # this frame is at the end of the trajectory, flow doesn't exist
                return None
            framestr2 = str(int(framestr) + flownum).zfill(len(framestr))
            return trajstr + '/' + datatype + '/' + framestr + '_' + framestr2 + '_flow.npy'

    def load_image(self, fn):
        img = cv2.imread(self.dataroot + '/' + fn, cv2.IMREAD_UNCHANGED)
        assert img is not None, "Error loading image {}".format(self.dataroot + '/' + fn)
        return img

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        if fn is None: 
            # print('return 0 for flow')
            return np.zeros((10,10,2),dtype=np.float32), np.zeros((10,10),dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow = np.load(self.dataroot + '/' + fn)
        mask = np.load(self.dataroot + '/' + fn.replace('flow.npy', 'mask.npy'))
        return flow, mask

    def load_depth(self, fn):
        depth = np.load(self.dataroot + '/' + fn)
        return depth

    def load_disparity(self, fn):
        depth = self.load_depth(fn)
        disp = 80.0/depth # hard coded
        return disp
