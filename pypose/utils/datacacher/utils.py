import torch

# def loadPretrain(model, preTrainModel):
#     preTrainDict = torch.load(preTrainModel)
#     model_dict = model.state_dict()
#     print 'preTrainDict:',preTrainDict.keys()
#     print 'modelDict:',model_dict.keys()
#     preTrainDict = {k:v for k,v in preTrainDict.items() if k in model_dict}
#     for item in preTrainDict:
#         print '  Load pretrained layer: ',item
#     model_dict.update(preTrainDict)
#     # for item in model_dict:
#     #   print '  Model layer: ',item
#     model.load_state_dict(model_dict)
#     return model

# from __future__ import division
# import torch
import math
import random
# from PIL import Image, ImageOps
import numpy as np
import numbers
import cv2

# ===== general functions =====

# Difference between ResizeData, RandomResizeCrop, DownscaleFlow
#   ResizeData: resize the data to a specific size
#   RandomResizeCrop: the resize factor is randomly generated
#   DownscaleFlow: down sample flow to 1/4 size
# Deprecate the support for non-seq data - 03/07/2022
# Deprecate the support for scale_disp parameter in resizing the data - 07/03/2022

KEY2DIM =  {'flow':3, 'img0':3, 'img1':3, 'img0_norm':3, 'img1_norm':3, \
            'intrinsic':3, 'fmask':2, 'disp0':2, 'disp1':2, 'depth0':2, 'depth1':2, \
            'flow_unc':2, 'depth0_unc':2} # data and data dimensions

class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def get_sample_dimention(sample):
    for kk in sample.keys():
        if kk in KEY2DIM: # for sequencial data
            h, w = sample[kk][0].shape[0], sample[kk][0].shape[1]
            return h, w
    assert False,"No image type in {}".format(sample.keys())

class RandomCrop(object):
    """Crops the given imgage(in numpy format) at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    The image should be in shape: (h, w) or (h, w, c)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        # self.key2dim = {'flow':3, 'img0':3, 'img1':3, 'intrinsic':3, 'fmask':2, 'disp0':2, 'disp1':2} # these data have 3 dimensions

    def __call__(self, sample):

        th, tw = self.size
        h, w = get_sample_dimention(sample)

        th = min(th, h)
        tw = min(tw, w)
        if w == tw and h == th:
            return sample
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for kk in sample.keys():
            if sample[kk] is None or (kk not in KEY2DIM):
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen): 
                datalist.append(sample[kk][k][y1:y1+th,x1:x1+tw,...])
            sample[kk] = datalist
        return sample

# TODO: make sure scale_disp depandency is clean
class CropCenter(object):
    """Crops the a sample of data (tuple) at center
    (TODO: deprecate the resizing) if the image size is not large enough, it will be first resized with fixed ratio
    if fix_ratio is False, w and h are resized separatedly
    (deprecated) if scale_w is given, w will be resized accordingly
    """

    def __init__(self, size, fix_ratio=True, scale_w=1.0, scale_disp=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fix_ratio = fix_ratio
        self.scale_w = scale_w
        # self.scale_disp = scale_disp

    def __call__(self, sample):

        th, tw = self.size
        hh, ww = get_sample_dimention(sample)
        if ww == tw and hh == th:
            return sample
        # import ipdb;ipdb.set_trace()
        # resize the image if the image size is smaller than the target size
        scale_h = max(1, float(th)/hh)
        scale_w = max(1, float(tw)/ww)
        if scale_h>1 or scale_w>1:
            if self.fix_ratio:
                scale_h = max(scale_h, scale_w)
                scale_w = max(scale_h, scale_w)
            w = int(round(ww * scale_w)) # w after resize
            h = int(round(hh * scale_h)) # h after resize
        else:
            w, h = ww, hh

        if self.scale_w != 1.0:
            scale_w = self.scale_w
            w = int(round(ww * scale_w))

        if scale_h != 1. or scale_w != 1.: # resize the data
            resizedata = ResizeData(size=(h, w), scale_disp=self.scale_disp)
            sample = resizedata(sample)

        x1 = int((w-tw)/2)
        y1 = int((h-th)/2)
        # import ipdb;ipdb.set_trace()
        for kk in sample.keys():
            if sample[kk] is None or (kk not in KEY2DIM):
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen): 
                datalist.append(sample[kk][k][y1:y1+th,x1:x1+tw,...])
            sample[kk] = datalist

        return sample

# TODO: make sure scale_disp depandency is clean
class ResizeData(object):
    """Resize the data in a dict
    """

    def __init__(self, size, fx=None, fy=None, scale_disp=False):
        if isinstance(size, numbers.Number):
            self.th, self.tw = int(size), int(size)
        else:
            self.th, self.tw = size
        # self.scale_disp = scale_disp
        if self.th <= 0 or self.tw <= 0:
            assert (fx is not None) and (fy is not None), "ResizeData size and fx/fy not specified!"
            self.scale_w, self.scale_h = fx, fy
            self.th, self.tw = 0, 0

    def __call__(self, sample):
        h, w = get_sample_dimention(sample)

        if self.th > 0 and self.tw > 0:
            self.scale_w = float(self.tw)/w
            self.scale_h = float(self.th)/h

        for kk in sample.keys():
            if sample[kk] is None or (kk not in KEY2DIM):
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen): 
                img = cv2.resize(sample[kk][k], (self.tw, self.th), 
                                 fx=self.scale_w, fy=self.scale_h, interpolation=cv2.INTER_LINEAR)
                if kk == 'flow':
                    img[...,0] = img[...,0] * self.scale_w
                    img[...,1] = img[...,1] * self.scale_h
                if kk.startswith('disp'): # and self.scale_disp:
                    img = img * self.scale_w
                datalist.append(img)
            sample[kk] = datalist

        sample['scale_w'] = np.array([self.scale_w ],dtype=np.float32)# used in e2e-stereo-vo

        return sample

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # import ipdb;ipdb.set_trace()
        for kk in sample.keys():
            if not kk in KEY2DIM:
                continue
            if KEY2DIM[kk] ==3: # for sequencial data
                data = np.stack(sample[kk], axis=0)
                data = data.transpose(0, 3, 1, 2) # frame x channel x h x w
            elif KEY2DIM[kk] ==2: # for sequencial data
                data = np.stack(sample[kk], axis=0)
                data = data[:,np.newaxis,:,:] # frame x channel x h x w

            data = data.astype(np.float32)
            sample[kk] = torch.from_numpy(data) # copy to make memory continuous

        return sample

class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
    This option should be before the to tensor
    """

    def __init__(self, mean, std, rgbbgr=False, keep_old=False):
        '''
        keep_old: keep both normalized and unnormalized data, 
        normalized data will be put under new key xxx_norm
        '''
        self.mean = mean
        self.std = std
        self.rgbbgr = rgbbgr
        self.keep_old = keep_old

    def __call__(self, sample):
        keys = list(sample.keys())
        for kk in keys:
            if kk.startswith('img0') or kk.startswith('img1'): # sample[kk] is a list, sample[kk][k]: h x w x 3
                seqlen = len(sample[kk])
                datalist = []
                for s in range(seqlen):
                    sample[kk][s] = sample[kk][s]/255.0
                    if self.rgbbgr:
                        img = sample[kk][s][...,[2,1,0]] # bgr2rgb
                    if self.mean is not None and self.std is not None:
                        img = np.zeros_like(sample[kk][s])
                        for k in range(3):
                            img[...,k] = (sample[kk][s][...,k] - self.mean[k])/self.std[k]
                    else:
                        img = sample[kk][s]
                    datalist.append(img)

                if self.keep_old:
                    sample[kk+'_norm'] = datalist
                else:
                    sample[kk] = datalist
        return sample

class RandomHSV(object):
    """
    Change the image in HSV space
    """

    def __init__(self, HSVscale=(6,30,30), random_random=0):
        '''
        random_random > 0: different images use different HSV x% of the original random HSV value
        '''
        self.Hscale, self.Sscale, self.Vscale = HSVscale
        self.random_random = random_random

    def additional_random_value(self,h,s,v):
        if self.random_random > 0: # add more noise to h s v
            hh = random.uniform(-1, 1) * self.random_random * self.Hscale + h
            ss = random.uniform(-1, 1) * self.random_random * self.Sscale + s
            vv = random.uniform(-1, 1) * self.random_random * self.Vscale + v
        else:
            hh, ss, vv = h, s, v 
        return int(hh),int(ss),int(vv)

    def __call__(self, sample):
        # change HSV
        h = random.uniform(-1,1) * self.Hscale
        s = random.uniform(-1,1) * self.Sscale
        v = random.uniform(-1,1) * self.Vscale

        for kk in sample.keys():
            if sample[kk] is None:
                continue
            if kk in {'img0', 'img1'}:
                seqlen = len(sample[kk])
                datalist = []
                for w in range(seqlen):
                    hh, ss, vv = self.additional_random_value(h,s,v)
                    imghsv = cv2.cvtColor(sample[kk][w], cv2.COLOR_BGR2HSV)
                    # import ipdb;ipdb.set_trace()
                    imghsv = imghsv.astype(np.int16)
                    imghsv[:,:,0] = np.clip(imghsv[:,:,0]+hh,0,255)
                    imghsv[:,:,1] = np.clip(imghsv[:,:,1]+ss,0,255)
                    imghsv[:,:,2] = np.clip(imghsv[:,:,2]+vv,0,255)
                    imghsv = imghsv.astype(np.uint8)
                    datalist.append(cv2.cvtColor(imghsv,cv2.COLOR_HSV2BGR))
                sample[kk] = datalist 
        return sample

class FlipFlow(object):
    """
    Flip up-down and left-right, change the flow value accordingly
    """

    def __init__(self):
        pass

    def __call__(self, sample):        
        flipud, fliplr = False, False
        if random.random()>0.5:
            flipud = True
        if random.random()>0.5:
            fliplr = True

        if not flipud and not fliplr:
            return sample

        for kk in sample.keys():
            if sample[kk] is None:
                continue
            seqlen = len(sample[kk])
            datalist = []
            if kk in {'img0', 'img1'}:
                for k in range(seqlen):
                    img = sample[kk][k]
                    if flipud:
                        img = np.flipud(img)
                    if fliplr:
                        img = np.fliplr(img)
                    datalist.append(img)
                if fliplr or flipud:
                    sample[kk] = datalist

            elif kk == 'flow':
                for k in range(seqlen):
                    img = sample['flow'][k]
                    if flipud:
                        img = np.flipud(img)
                        img[:,:,1] = -img[:,:,1]
                    if fliplr:
                        img = np.fliplr(img)
                        img[:,:,0] = -img[:,:,0]
                    datalist.append(img)
                sample['flow'] = datalist

            elif kk == 'fmask':
                for k in range(seqlen):
                    img = sample['fmask'][k]
                    if flipud:
                        img = np.flipud(img)
                    if fliplr:
                        img = np.fliplr(img)
                    datalist.append(img)
                sample['fmask'] = datalist

        return sample

class FlowStereoNormalization(object):
    """
    Normalize the flow or stereo
    """

    def __init__(self, norm_factor, mod):
        self.norm_factor = norm_factor
        self.mod = mod

    def __call__(self, sample):        

        seqlen = len(sample[self.mod])
        datalist = []
        for k in range(seqlen):
            img = sample[self.mod][k] * self.norm_factor
            datalist.append(img)
        sample[self.mod] = datalist

        return sample

# ===== transform for stereo =====
# deprecated for no obvious improvement
# class CombineLR(object):
#     '''
#     combine the left and right images for stereo
#     '''
#     def __call__(self, sample):
#         leftImg, rightImg = sample['img0'], sample['img1']
#         rbgs = torch.cat((leftImg, rightImg),dim=0)
#         return { 'rgbs':  rbgs, 'disp0': sample['disp0']}

class FlipStereo(object):
    """
    Flip depth up-down
    """
    def __init__(self):
        pass

    def __call__(self, sample):        
        if random.random()>0.5:
            for kk in sample.keys():
                if sample[kk] is None:
                    continue
                if kk.startswith('img') or kk.startswith('disp'): 
                    seqlen = len(sample[kk])
                    datalist = []
                    for k in range(seqlen):
                        datalist.append(np.flipud(sample[kk][k]))
                    sample[kk] = datalist
        return sample

class RandomRotate(object):
    """
    Randomly rotate the right image by a small angle
    """
    def __init__(self, maxangle=1.0):
        self.maxangle = maxangle

    def rotate_image(self, image, angle):
        rotate_center = (np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0]))#tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        # print('center: {}, angle {}'.format(rotate_center, angle))
        return result

    def __call__(self, sample):
        if 'img1' not in sample.keys():
            print("RandomRotate Error: No right image found!")
            return sample
        if random.random()>0.5:
            seqlen = len(sample['img1'])
            datalist = []
            for k in range(seqlen):
                rotangle = (np.random.random() *2 -1) * self.maxangle
                img_rot = self.rotate_image(sample['img1'][k], rotangle)
                datalist.append(img_rot)
            sample['img1'] = datalist
        return sample

# ===== transform for flowvo ======
def generate_random_scale_crop(h, w, target_h, target_w, scale_base, keep_center, fix_ratio):
    '''
    Randomly generate scale and crop params
    H: input image h
    w: input image w
    target_h: output image h
    target_w: output image w
    scale_base: max scale up rate
    keep_center: crop at center
    fix_ratio: scale_h == scale_w
    '''
    scale_w = random.random() * (scale_base - 1) + 1
    if fix_ratio:
        scale_h = scale_w
    else:
        scale_h = random.random() * (scale_base - 1) + 1

    crop_w = int(math.ceil(target_w/scale_w)) # ceil for redundancy
    crop_h = int(math.ceil(target_h/scale_h)) # crop_w * scale_w > w

    if keep_center:
        x1 = int((w-crop_w)/2)
        y1 = int((h-crop_h)/2)
    else:
        x1 = random.randint(0, w - crop_w)
        y1 = random.randint(0, h - crop_h)

    return scale_w, scale_h, x1, y1, crop_w, crop_h

class Tartan2Kitti2(object):
    def __init__(self):
        focal_t = 320.0
        cx_t = 320.0
        cy_t = 240
        focal_k = 707.0912
        cx_k = 601.8873
        cy_k = 183.1104


        self.th = 370
        self.tw = 1226

        self.sx = (0-cx_k)/focal_k*focal_t+cx_t
        self.sy = (0-cy_k)/focal_k*focal_t+cy_t

        self.step = focal_t/ focal_k
        self.scale = focal_k/ focal_t

    def __call__(self, flownp):
        # import ipdb;ipdb.set_trace()
        res = np.zeros((self.th, self.tw, flownp.shape[-1]), dtype=np.float32)
        thh = 0.
        for hh in range(self.th):
            tww = 0.
            for ww in range(self.tw):
                res[hh, ww, :] = bilinear_interpolate(flownp, self.sy+thh, self.sx+tww)
                tww += self.step
            thh += self.step

        res = res * self.scale

        return res

class Tartan2Kitti(object):
    def __init__(self):
        pass

    def __call__(self, flownp):
        # import ipdb;ipdb.set_trace()
        flowcrop = flownp[157:325 ,48:603, :]
        flowcrop = cv2.resize(flowcrop, (1226,370), interpolation=cv2.INTER_LINEAR)
        # scale the flow
        flowcrop[:,:,0] = flowcrop[:,:,0] * (2.209)
        flowcrop[:,:,1] = flowcrop[:,:,1] * (2.202)

        return flowcrop


# ========= end-to-end flow and vo ==========
# TODO: make sure scale_disp related dependancy is clean
class RandomResizeCrop(object):
    """
    Random scale to cover continuous focal length
    Due to the tartanair focal is already small, we only up scale the image
    """

    def __init__(self, size, max_scale=2.5, keep_center=False, fix_ratio=False, scale_disp=False):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        (depracated) scale_disp: when training the stereovo, disparity represents depth, which is not scaled with resize 
        '''
        if isinstance(size, numbers.Number):
            self.target_h = int(size)
            self.target_w = int(size)
        else:
            self.target_h = size[0]
            self.target_w = size[1]

        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        # self.scale_disp = scale_disp

        self.scale_base = max_scale #self.max_focal /self.tartan_focal

    def crop_resize(self, img, scale_w, scale_h, x1, y1, crop_w, crop_h):
        img = img[y1:y1+crop_h, x1:x1+crop_w]
        img = cv2.resize(img, (0,0), fx=scale_w, fy=scale_h, interpolation=cv2.INTER_LINEAR)
        # Note opencv reduces the last dimention if it is one
        img = img[:self.target_h,:self.target_w]
        return img

    def __call__(self, sample): 
        h, w = get_sample_dimention(sample)
        self.target_h = min(self.target_h, h)
        self.target_w = min(self.target_w, w)

        scale_w, scale_h, x1, y1, crop_w, crop_h = generate_random_scale_crop(h, w, self.target_h, self.target_w, 
                                                    self.scale_base, self.keep_center, self.fix_ratio)

        for kk in sample:
            if kk not in KEY2DIM: 
                continue
            seqlen = len(sample[kk])
            datalist = []
            for k in range(seqlen): 
                img = self.crop_resize(sample[kk][k], scale_w, scale_h, x1, y1, crop_w, crop_h)
                if kk == 'flow':
                    img[...,0] = img[...,0] * scale_w
                    img[...,1] = img[...,1] * scale_h
                if kk.startswith('disp'): # and self.scale_disp:
                    img = img * scale_w
                datalist.append(img)
            sample[kk] = datalist
    
        # if not self.scale_disp:
        sample['scale_w'] = np.array([scale_w ],dtype=np.float32)# used in e2e-stereo-vo

        return sample


class DownscaleFlow(object):
    """
    Scale the flow and mask to a fixed size
    This function won't resize the RGBs
    flow/disp values will NOT be changed

    """
    def __init__(self, scale=4):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        '''
        self.downscale = 1.0/scale
        # self.key2dim = {'flow':3, 'intrinsic':3, 'fmask':2, 'disp0':2, 'disp1':2} # these data have 3 dimensions

    def __call__(self, sample): 
        if self.downscale==1:
            return sample

        # import ipdb;ipdb.set_trace()
        for key in sample.keys():
            if key in {'flow','intrinsic','fmask','disp0','depth0'}:
                imgseq = []
                for k in range(len(sample[key])):
                    imgseq.append(cv2.resize(sample[key][k], 
                        (0, 0), fx=self.downscale, fy=self.downscale, interpolation=cv2.INTER_LINEAR))
                sample[key] = imgseq

        return sample

# ==== stereo vo ====
# TODO: adapt to seqloader
class RandomScaleDispMotion(object):
    """
    Scaling the disparity and translation at the same time will still hold
    """

    def __init__(self, MAX_MOTION=2.0, MAX_SCALE=5.0, disp_norm=20.0):
        '''
        '''
        self.MAX_MOTION = MAX_MOTION / 0.13 # hard code the motion normalization value
        self.MAX_SCALE = MAX_SCALE
        self.disp_norm = disp_norm
        # mean disp and max disp should be large enough, otherwise do not do this augmentation
        # disp_mean_unnorm: 16, disp_max_unnorm: 60
        # dist_mean > 5m, dist_min > 1.33m  
        self.disp_mean_thresh = 16.0/self.disp_norm
        self.disp_max_thresh = 60.0/self.disp_norm

    def __call__(self, sample):
        if ('motion' not in sample) or ('disp0' not in sample):
            print('ERROR!!')
            return sample

        disp = sample['disp0']
        disp_mean = disp.mean() 
        disp_max = disp.max()
        if disp_mean < self.disp_mean_thresh and disp_max < self.disp_max_thresh: # everything is already far away, do not scale further 
            # print('TOO FAR!!')
            return sample

        motion = sample['motion']
        motion_norm = np.linalg.norm(motion[:3])
        motion_max_scale = self.MAX_MOTION/motion_norm
        motion_max_scale = max(1, min(self.MAX_SCALE, motion_max_scale))
        if motion_max_scale <= 1: # motion is already big
            # print('--big motion!!')
            return sample

        motion_scale = abs(random.gauss(0, 0.5) * (motion_max_scale-1)) + 1
        # print('{}, {}, {}, {}'.format(disp_mean, disp_max, motion_norm, motion_scale))
        # scale the data
        sample['motion'][:3] = sample['motion'][:3] * motion_scale
        sample['disp0'] = sample['disp0'] / motion_scale

        return sample

# TODO: adapt to seqloader
class StaticMotion(object):
    """
    Randomly turn one sample into static case
    """

    def __init__(self, Rate=0.01):
        '''
        '''
        self.Rate = Rate

    def __call__(self, sample):
        if random.random() > self.Rate:
            return sample

        if 'motion' in sample:
            sample['motion'] = np.array([0.,0.,0.,0.,0.,0.]).astype(np.float32)
        if 'flow' in sample:
            sample['flow'] = np.zeros_like(sample['flow']).astype(np.float32)
        if 'img0n' in sample and 'img0' in sample:
            sample['img0n'] = sample['img0'].copy()

        return sample

class RandomUncertainty(object):
    '''
    Generate random patches of noisy region and corresponding uncertainty
    '''
    def __init__(self, patchnum=5, data_max_bias=5.0, data_noise_scale=1.0): #NoiseSigma, NoiseMean
        '''
        data_max_bias: max bias value add to data
        data_noise_scale: the scale of white noise added to data
        '''
        self.patchnum = patchnum
        self.data_max_bias = data_max_bias
        self.data_noise_scale = data_noise_scale

    def random_patch(self, w, h, maxratio, minratio=None):
        maxw, maxh = int(w*maxratio), int(h*maxratio)
        if minratio is None:
            minw, minh = 1, 1
        else:
            minw, minh = int(w*minratio)+1, int(h*minratio)+1
        rw, rh = random.randint(minw, maxw), random.randint(minh, maxh)
        sy, sx = random.randint(0, w-rw), random.randint(0, h-rh)
        return sx, sy, rh, rw

    def generate_noise(self, rh, rw, c):
        '''
        '''
        # generate noise for the data
        # import ipdb;ipdb.set_trace()
        bias_base = np.random.rand(c)*2-1 # (-1, 1) x c
        bias = bias_base * self.data_max_bias # (-b, b) x c
        whitenoise_base = np.random.randn(rh, rw, c) # (-1, 1)
        whitenoise = whitenoise_base * self.data_noise_scale # (-n, n)
        data_noise = bias.reshape(1,1,c) + whitenoise 
        # generate mask according to the data noise
        mask_whitenoise = np.abs(np.random.randn(rh, rw)) # (0, ~2)
        mask = np.clip(mask_whitenoise-1, 0, 1 ) # (0, m * mask_base)

        return data_noise, mask

    def random_mask_flow(self, img, mask, maxratio=0.5, maxholes=5):
        '''
        img: h x w x c, c could be 1
        '''
        # import ipdb;ipdb.set_trace()
        h, w, c = img.shape
        # mask = np.random.rand(h,w) * 2 - 4 # generate the background ranging from (-4, -2)
        img_noise = img.copy()

        sx, sy, rh, rw = self.random_patch(w, h, maxratio, minratio=0.1)
        data_noise, mask_noise = self.generate_noise(rh, rw, c)
        img_noise[sx:sx+rh, sy:sy+rw] = img_noise[sx:sx+rh, sy:sy+rw] + data_noise
        mask[sx:sx+rh, sy:sy+rw] = mask_noise

        # add a few holes 
        for k in range(random.randint(0, maxholes)): 
            hx, hy, hh, hw = self.random_patch(rw, rh, maxratio, minratio=0.1)
            hx, hy = sx + hx, sy + hy
            img_noise[hx:hx+hh, hy:hy+hw] = img[hx:hx+hh, hy:hy+hw]
            mask[hx:hx+hh, hy:hy+hw] = np.clip(1 - np.random.randn(hh,hw) * 0.02, 0, 1)

        return img_noise, mask

    def random_mask_depth(self, img, mask, maxratio=0.5, maxholes=5):
        '''
        img: h x w x c, c could be 1
        '''
        # import ipdb;ipdb.set_trace()
        h, w, c = img.shape
        # mask = np.random.rand(h,w) * 2 - 4 # generate the background ranging from (-4, -2)
        img_noise = img.copy()
        imgmin = img.min()
        sx, sy, rh, rw = self.random_patch(w, h, maxratio, minratio=0.1)
        data_noise, mask_noise = self.generate_noise(rh, rw, c)
        img_noise[sx:sx+rh, sy:sy+rw] = img_noise[sx:sx+rh, sy:sy+rw] + data_noise
        img_noise = np.clip(img_noise, imgmin, None)
        
        mask[sx:sx+rh, sy:sy+rw] = mask_noise

        # add a few holes 
        for k in range(random.randint(0, maxholes)): 
            hx, hy, hh, hw = self.random_patch(rw, rh, maxratio, minratio=0.1)
            hx, hy = sx + hx, sy + hy
            img_noise[hx:hx+hh, hy:hy+hw] = img[hx:hx+hh, hy:hy+hw]
            mask[hx:hx+hh, hy:hy+hw] = np.clip(1 - np.random.randn(hh,hw) * 0.02, 0, 1)

        return img_noise, mask

    def __call__(self, sample, ):

        if 'flow' in sample:
            dataseq = []
            uncseq = []
            for k in range(len(sample['flow'])):
                flow = sample['flow'][k]
                h, w, c = flow.shape
                unc_mask = np.clip(1 - np.random.randn(h,w) * 0.02, 0, 1)  # generate the certain background (0.98, 1)
                for k in range(random.randint(0,self.patchnum)):
                    flow, unc_mask = self.random_mask_flow(flow, unc_mask)
                dataseq.append(flow)
                uncseq.append(unc_mask)
            sample['flow'] = dataseq
            sample['flow_unc'] = uncseq

        if 'depth0' in sample:
            dataseq = []
            uncseq = []
            for k in range(len(sample['depth0'])):
                depth = sample['depth0'][k]
                h, w = depth.shape
                unc_mask = np.clip(1 - np.random.randn(h,w) * 0.02, 0, 1)  # generate the certain background (0.98, 1)
                depth = depth.reshape(h,w,1)
                for k in range(random.randint(0,self.patchnum)):
                    depth, unc_mask = self.random_mask_depth(depth, unc_mask)
                dataseq.append(depth.squeeze(-1))
                uncseq.append(unc_mask)
            sample['depth0'] = dataseq
            sample['depth0_unc'] = uncseq

        return sample

# ===== IMU ======
# accel = imudata[:, :3]
# gyro = imudata[:, 3:6]
# vel_world = imudata[:, 6:9]
# angles_world = imudata[:, 9:12]

class IMUNormalization(object):
    def __init__(self):
        '''
        Normalization should be called after adding the noise
        '''
        from .imu_noise import vel_mean, vel_dev, accel_mean, accel_dev, gyro_mean, gyro_dev, angles_mean, angles_dev, angles_6dof_mean, angles_6dof_dev, accel_nograv_mean, accel_nograv_dev
        self.vel_mean = vel_mean.reshape((1,3))
        self.vel_dev = vel_dev.reshape((1,3))
        self.accel_mean = accel_mean.reshape((1,3))
        self.accel_dev = accel_dev.reshape((1,3))
        self.gyro_mean = gyro_mean.reshape((1,3))
        self.gyro_dev = gyro_dev.reshape((1,3))
        self.angles_mean = angles_mean.reshape((1,3))
        self.angles_dev = angles_dev.reshape((1,3))
        self.angles_6dof_mean = angles_6dof_mean.reshape((1,6))
        self.angles_6dof_dev = angles_6dof_dev.reshape((1,6))
        self.accel_nograv_mean = accel_nograv_mean.reshape((1,3))
        self.accel_nograv_dev = accel_nograv_dev.reshape((1,3))

    def __call__(self, sample):
        for key in ['imu', 'imu_noise']:
            if key in sample:
                imudata = sample[key]
                accel = imudata[:, :3]
                gyro = imudata[:, 3:6]
                vel_world = imudata[:, 6:9]
                angles_world = imudata[:, 9:12]
                accel_nograv = imudata[:, 12:15]
                # angles_world_6dof = imudata[:, 12:18]
                accel = (accel - self.accel_mean) / self.accel_dev
                gyro = (gyro - self.gyro_mean) / self.gyro_dev
                vel_world = (vel_world - self.vel_mean) / self.vel_dev
                angles_world = (angles_world - self.angles_mean) / self.angles_dev
                accel_nograv = (accel_nograv - self.accel_nograv_mean) / self.accel_nograv_dev
                # angles_world_6dof = (angles_world_6dof - self.angles_6dof_mean) / self.angles_6dof_dev
                sample[key] = np.concatenate([accel, gyro, vel_world, angles_world, accel_nograv], axis=1) # N x 12
        return sample

# from .imu_noise import add_realsense_noise
        
# class IMUNoise(object):
#     def __init__(self, noiselevel=0.0):
#         self.noiselevel = noiselevel

#     def __call__(self, sample): 
#         # import ipdb;ipdb.set_trace()
#         imudata = sample['imu']
#         accel = imudata[:, :3]
#         gyro = imudata[:, 3:6]
#         accel_noise, gyro_noise = add_realsense_noise(accel, gyro, self.noiselevel) 
#         imudata_noise = np.concatenate((accel_noise, gyro_noise, imudata[:, 6:]), axis=1)
#         sample['imu_noise'] = imudata_noise
#         return sample

# def imu_denormalize(accel=None, gyro=None, vel=None, angle=None):
#     '''
#     undo the IMUNormalization for loss comparison
#     '''
#     from imu_noise import vel_mean, vel_dev, accel_mean, accel_dev, gyro_mean, gyro_dev, angles_mean, angles_dev
#     if len(gyro.shape) == 2:
#         newshape = (1,3)
#     elif len(gyro.shape) == 3:
#         newshape = (1,1,3)
#     else:
#         print('imu_denormalize: input shape not supported')
#         return accel, gyro, vel, angle
#     if angle is not None:
#         if angle.shape[-1]==6:
#             newshape2 = newshape[:-1]+(6,)
#             from imu_noise import angles_6dof_mean, angles_6dof_dev
#             angle = (angle*angles_6dof_dev.reshape(newshape2)) + angles_6dof_mean.reshape(newshape2)
#         else:
#             angle = (angle*angles_dev.reshape(newshape)) + angles_mean.reshape(newshape)
#     if vel is not None:
#         vel = (vel*vel_dev.reshape(newshape)) + vel_mean.reshape(newshape)
#     if gyro is not None:
#         gyro = (gyro*gyro_dev.reshape(newshape)) + gyro_mean.reshape(newshape)
#     if accel is not None:
#         accel = (accel*accel_dev.reshape(newshape)) + accel_mean.reshape(newshape)

#     return accel, gyro, vel, angle

def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    if mean is not None and std is not None:
        for t, m, s in zip(tensImg, mean, std):
            t.mul_(s).add_(m) 
    tensImg = np.clip(tensImg * float(255),0,255)
    # undo transpose
    tensImg = (tensImg.numpy().transpose(1,2,0)).astype(np.uint8)
    return tensImg

def bilinear_interpolate(img, h, w):
    # assert round(h)>=0 and round(h)<img.shape[0]
    # assert round(w)>=0 and round(w)<img.shape[1]

    h0 = int(math.floor(h))
    h1 = h0 + 1
    w0 = int(math.floor(w))
    w1 = w0 + 1

    a = h - h0 
    b = w - w0

    h0 = max(h0, 0)
    w0 = max(w0, 0)
    h1 = min(h1, img.shape[0]-1)
    w1 = min(w1, img.shape[1]-1)

    A = img[h0,w0,:]
    B = img[h1,w0,:]
    C = img[h0,w1,:]
    D = img[h1,w1,:]

    res = (1-a)*(1-b)*A + a*(1-b)*B + (1-a)*b*C + a*b*D

    return res 

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr

def visdepth(disp, scale=3):
    res = np.clip(disp*scale, 0, 255).astype(np.uint8)
    res = cv2.applyColorMap(res, cv2.COLORMAP_JET)
    # res = np.tile(res[:,:,np.newaxis], (1, 1, 3))
    return res

def dataset_intrinsics(dataset='tartanair', calibfile=None):
    if dataset == 'kitti':
        # focalx, focaly, centerx, centery = 707.0912, 707.0912, 601.8873, 183.1104
        with open(calibfile, 'r') as f:
            lines = f.readlines()
        cam_intrinsics = lines[2].strip().split(' ')[1:]
        focalx, focaly, centerx, centery = float(cam_intrinsics[0]), float(cam_intrinsics[5]), float(cam_intrinsics[2]), float(cam_intrinsics[6])
    elif dataset == 'euroc':
        # focalx, focaly, centerx, centery = 355.6358642578, 417.1617736816, 362.2718811035, 249.6590118408
        focalx, focaly, centerx, centery = 458.6539916992, 457.2959899902, 367.2149963379, 248.3750000000

    elif dataset == 'tartanair':
        focalx, focaly, centerx, centery = 320.0, 320.0, 320.0, 240.0

    elif dataset == 'realsense':
        focalx, focaly, centerx, centery = 379.2695007324, 379.4417419434, 317.5869721176, 239.5056236642
    else:
        return None
    return focalx, focaly, centerx, centery

def make_intrinsics_layer(intrinsics):
    w, h, fx, fy, ox, oy = intrinsics
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww, hh)).transpose(1,2,0)
    return intrinsicLayer

# uncompress the data
def flow16to32(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:,:,:2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    mask8 = flow16[:,:,2].astype(np.uint8)
    return flow32, mask8

# mask = 1  : CROSS_OCC 
#      = 10 : SELF_OCC
#      = 100: OUT_OF_FOV
#      = 200: OVER_THRESHOLD
def flow32to16(flow32, mask8):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_16b = (flow_32b * 64) + 32768  
    '''
    # mask flow values that out of the threshold -512.0 ~ 511.984375
    mask1 = flow32 < -512.0
    mask2 = flow32 > 511.984375
    mask = mask1[:,:,0] + mask2[:,:,0] + mask1[:,:,1] + mask2[:,:,1]
    # convert 32bit to 16bit
    h, w, c = flow32.shape
    flow16 = np.zeros((h, w, 3), dtype=np.uint16)
    flow_temp = (flow32 * 64) + 32768
    flow_temp = np.clip(flow_temp, 0, 65535)
    flow_temp = np.round(flow_temp)
    flow16[:,:,:2] = flow_temp.astype(np.uint16)
    mask8[mask] = 200
    flow16[:,:,2] = mask8.astype(np.uint16)

    return flow16

def depth_rgba_float32(depth_rgba):
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


def depth_float32_rgba(depth):
    '''
    depth: float32, h x w
    store depth in uint8 h x w x 4
    and use png compression
    '''
    depth_rgba = depth[...,np.newaxis].view("<u1")
    return depth_rgba

def per_frame_scale_alignment(gt_motions, est_motions):
    dist_gt = np.linalg.norm(gt_motions[:,:3], axis=1)
    # scale the output frame by frame
    motions_scale = est_motions.copy()
    dist = np.linalg.norm(motions_scale[:,:3],axis=1)
    scale_gt = dist_gt/dist
    motions_scale[:,:3] = est_motions[:,:3] * scale_gt.reshape(-1,1)

    return motions_scale


import re
 
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode('ascii') == 'PF':
        color = True
    elif header.decode('ascii') == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('ascii'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale

def read_flo(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

if __name__ == '__main__':
    # testflow = np.random.rand(50,30,4).astype(np.float32) * 5
    # randintrinsic = RandomIntrinsic(minlen=10)
    # cropflow = randintrinsic.__call__(testflow)

    # img = cv2.imread('/home/amigo/tmp/data/tartan/amusement/Data_fast/CP001/image_left/000000_left.png') 
    # randomRotate = RandomRotate(maxangle=1.0)
    # sample={'img1':[img]}
    # for k in range(1000):
    #     # import ipdb;ipdb.set_trace()
    #     imgrot = randomRotate(sample) #randomRotate.rotate_image(img, 1.0)
    #     cv2.imshow('img', imgrot['img1'][0])
    #     cv2.waitKey(1)

    # flow0 = np.load('/home/amigo/tmp/data/tartan/amusement/Data_fast/OP001/flow/000000_000001_flow.npy') 
    # flow1 = np.load('/home/amigo/tmp/data/tartan/amusement/Data_fast/OP001/flow/000010_000011_flow.npy')
    # uncertanty = RandomUncertainty(data_max_bias=10, data_noise_scale=2)
    # for k in range(1000):
    #     sample={'flow':[flow0.copy(), flow1.copy()]}
    #     # import ipdb;ipdb.set_trace()
    #     sample_unc = uncertanty(sample) #randomRotate.rotate_image(img, 1.0)
    #     flowdisp =  np.concatenate((visflow(sample_unc['flow'][0]),  visflow(sample_unc['flow'][1])),axis=1)
    #     uncdisp = np.concatenate((visdepth(sample_unc['flow_unc'][0], scale=200),visdepth(sample_unc['flow_unc'][1], scale=200)),axis=1)
    #     cv2.imshow('img', np.concatenate((flowdisp, uncdisp),axis=0))
    #     cv2.waitKey(0)
    #     import ipdb;ipdb.set_trace()

    depth0 = np.load('/home/amigo/tmp/data/tartan/amusement/Data_fast/OP001/depth_left/000001_left_depth.npy') 
    depth1 = np.load('/home/amigo/tmp/data/tartan/amusement/Data_fast/OP001/depth_left/000011_left_depth.npy')
    uncertanty = RandomUncertainty(data_max_bias=10, data_noise_scale=2)
    for k in range(1000):
        sample={'depth0':[depth0.copy(), depth1.copy()]}
        # import ipdb;ipdb.set_trace()
        sample_unc = uncertanty(sample) #randomRotate.rotate_image(img, 1.0)
        depthdisp =  np.concatenate((visdepth(80./sample_unc['depth0'][0]),  visdepth(80./sample_unc['depth0'][1])),axis=1)
        uncdisp = np.concatenate((visdepth(sample_unc['depth0_unc'][0], scale=200),visdepth(sample_unc['depth0_unc'][1], scale=200)),axis=1)
        cv2.imshow('img', np.concatenate((depthdisp, uncdisp),axis=0))
        cv2.waitKey(0)
        # import ipdb;ipdb.set_trace()
