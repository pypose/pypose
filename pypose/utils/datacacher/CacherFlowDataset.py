import cv2
import numpy as np
from .CacherDatasetBase import CacherDatasetBase
from .utils import readPFM, read_flo


class CacherSintelDataset(CacherDatasetBase):

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):
        if datatype == 'img0':
            return trajstr + '/frame_' + framestr + '.png'
        if datatype == 'flow':
            if ind_inv <= 1: # this frame is at the end of the trajectory, flow doesn't exist
                return None
            trajstr = trajstr.replace('final', 'flow').replace('clean','flow') # hard code
            return trajstr + '/frame_' + framestr + '.flo'

    def load_image(self, fn):
        img = cv2.imread(self.dataroot + '/' + fn)
        assert img is not None, "Error loading image {}".format(self.dataroot + '/' + fn)
        return img[:,:,:3]

    def load_flow(self, fn):
        if fn is None: 
            return np.zeros((10,10,2),dtype=np.float32), np.zeros((10,10),dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow = read_flo(self.dataroot + '/' + fn)
        return flow, None

class CacherChairsDataset(CacherDatasetBase):

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):
        # Note: Chairs is different from other dataset
        #       The data is not in sequence
        if datatype == 'img0':
            return trajstr + '/' + framestr + '.ppm'
        elif datatype == 'flow':
            if ind_inv <= 1: # this frame is at the end of the trajectory, flow doesn't exist
                return None
            return trajstr + '/' + framestr.replace('_img1','_flow') + '.flo'
        assert False, "Unsupport type {}".format(datatype)

    def load_image(self, fn):
        img = cv2.imread(self.dataroot + '/' + fn)
        assert img is not None, "Error loading image {}".format(self.dataroot + '/' + fn)
        return img[:,:,:3]

    def load_flow(self, fn):
        if fn is None: 
            return np.zeros((10,10,2),dtype=np.float32), np.zeros((10,10),dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow = read_flo(self.dataroot + '/' + fn)
        return flow, None

class CacherFlyingDataset(CacherDatasetBase):

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):
        if datatype == 'img0':
            return trajstr + '/' + framestr + '.png'
        if datatype == 'flow':
            if ind_inv <= 1: # this frame is at the end of the trajectory, flow doesn't exist
                return None
            trajstr = trajstr.replace('frames_finalpass/','optical_flow/') # hard code
            trajstr = trajstr.replace('frames_cleanpass/','optical_flow/') # hard code
            # if self.flowinv: # hard code
            #     if 'left' in trajstr:
            #         trajstr = trajstr.replace('left', 'into_past/left')
            #         datapathlist.append(trajstr + '/OpticalFlowIntoPast_' + framestr + '_L.pfm')
            #     elif 'right' in trajstr:
            #         trajstr = trajstr.replace('right', 'into_past/right')
            #         datapathlist.append(trajstr + '/OpticalFlowIntoPast_' + framestr + '_R.pfm')
            # else:
            if 'left' in trajstr:
                trajstr = trajstr.replace('left', 'into_future/left')
                return trajstr + '/OpticalFlowIntoFuture_' + framestr + '_L.pfm'
            elif 'right' in trajstr:
                trajstr = trajstr.replace('right', 'into_future/right')
                return trajstr + '/OpticalFlowIntoFuture_' + framestr + '_R.pfm'

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        if fn is None: 
            return np.zeros((10,10,2),dtype=np.float32), np.zeros((10,10),dtype=np.uint8) # return an arbitrary shape because it will be resized later
        flow, _ = readPFM(self.dataroot + '/' + fn)
        assert flow is not None, "Error loading flow {}".format(self.dataroot + '/' + fn)
        return flow[:,:,:2], flow[:,:,2]

    def load_image(self, fn):
        img = cv2.imread(self.dataroot + '/' + fn)
        assert img is not None, "Error loading image {}".format(self.dataroot + '/' + fn)
        return img


if __name__ == '__main__':

    import time
    from .utils import visflow, RandomResizeCrop, RandomHSV, ToTensor, Normalize, tensor2img, Compose, FlipFlow
    from .data_roots import *

    normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    randResizeCrop =  RandomResizeCrop(size=(400, 400), keep_center=True, fix_ratio=False)
    transformlist = [ randResizeCrop, RandomHSV((10,80,80), random_random=0.5), FlipFlow(), normalize, ToTensor() ] # 

    datafile = 'data/hm01_flow.txt'

    # dataset_term = datafile.split('/')[-1].split('.txt')[0].split('_')[0]
    # dataroot_list = FLOW_DR[dataset_term]['local']

    # flowDataset = TartanFlowDataset(datafile, 
    #                                 imgCPrefix='', flowCPrefix='',
    #                                 transform=Compose(transformlist), has_mask=False)
    datafile = 'data/flyingchairs_local.txt'
    flowDataset = ChairsFlowDataset(datafile, 
                                    imgdir='/home/wenshan/tmp/data/flyingchairs', flowdir='/home/wenshan/tmp/data/flyingchairs',
                                    transform=Compose(transformlist), \
                                    flow_norm = 1., has_mask=False)
    
    for k in range(0,len(flowDataset),1):
        sample = flowDataset[k]
        flownp = sample['flow'].squeeze(0).numpy()
        flownp = flownp / flowDataset.flow_norm

        # flowmask = sample['fmask'].squeeze()

        # import ipdb;ipdb.set_trace()
        flownp = flownp.transpose(1,2,0)
        flowvis = visflow(flownp)
        # flowvis[flowmask>128,:] = 0
        img1 = sample['img0'][0,:,:,:]
        img2 = sample['img0'][1,:,:,:]
        img1 = tensor2img(img1, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        img2 = tensor2img(img2, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

        imgdisp = np.concatenate((img1, img2 ,flowvis), axis=0)
        imgdisp = cv2.resize(imgdisp,(0,0),fx=0.5,fy=0.5)
        cv2.imshow('img',imgdisp)
        cv2.waitKey(0)

        # # warp image2 to image1
        # warp21 = np.zeros_like(img1)
        # warp12 = np.zeros_like(img1)
        # for h in range(img1.shape[0]):
        #     for w in range(img1.shape[1]):
        #         th = h + flownp[h,w,1] 
        #         tw = w + flownp[h,w,0] 
        #         if round(th)>=0 and round(th)<img1.shape[0] and round(tw)>=0 and round(tw)<img1.shape[1]:
        #             warp21[h,w,:] = bilinear_interpolate(img2, th, tw).astype(np.uint8)
        #             warp12[int(round(th)), int(round(tw)), :] = img1[h,w,:]

        # # # import ipdb;ipdb.set_trace()
        # diff = warp21.astype(np.float32) - img1.astype(np.float32)
        # diff = np.abs(diff)
        # print 'diff:', diff.mean()
        # diff = diff.astype(np.uint8)
        # con1 = np.concatenate((img1,warp21,flowvis),axis=0)
        # con2 = np.concatenate((img2, warp12,diff), axis=0)
        # imgdisp = np.concatenate((con1, con2), axis=1)
        # imgdisp = cv2.resize(imgdisp,(0,0),fx=0.5,fy=0.5)
        # cv2.imshow('img',imgdisp)
        # cv2.waitKey(0)

