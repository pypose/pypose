import torch 
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import math
import random


def get_batch_size(data):
    if isinstance(data, list):
        return len(data)
    else:
        return data.shape[0]

class RCR(nn.Module):
    def __init__(self, size, max_scale=2.5, keep_center=False, fix_ratio=False, fix_scale=None):
        '''
        size: output frame size, this should be NO LARGER than the input frame size! 
        fix_scale = (scale_h, scale_w): no randomness in the scaling if fix_scale is given

        - The input can be: batch x c x h x w, or batch x seq x c x h x w, 
          all the modalities in one seq will use same random parameters
          different seqs will use different parameters
        - Note: img, disp, and flow are handled separately, because disp and flow value will be changed
                for depth, intrinsics and other, they can be passed in as 'img', and concatenated in the channel/seq
        - When max_scale == 1, it becomes random crop
        - When max_sacle == 1 and keep_center == True, it becomes center crop
        '''
        super(RCR, self).__init__()
        self.target_h, self.target_w = size

        self.keep_center = keep_center
        self.fix_ratio = fix_ratio
        self.fix_scale = fix_scale

        self.scale_base = max_scale #self.max_focal /self.tartan_focal
        assert max_scale>=1, "Random cropping scale: scale factor {} should be larger than 1! ".format(max_scale)

    def generate_random_scale_crop(self, h, w, target_h, target_w):
        '''
        Randomly generate scale and crop params
        H: input image h
        w: input image w
        target_h: output image h
        target_w: output image w
        scale_base: max scale up rate
        keep_center: crop at center
        fix_ratio: scale_h == scale_w
        fix_scale: use the given scale
        '''
        if self.fix_scale is None:
            scale_w = random.random() * (self.scale_base - 1) + 1
            if self.fix_ratio:
                scale_h = scale_w
            else:
                scale_h = random.random() * (self.scale_base - 1) + 1
        else:
            scale_h, scale_w = self.fix_scale

        crop_w = int(math.ceil(target_w/scale_w)) # ceil for redundancy
        crop_h = int(math.ceil(target_h/scale_h)) # crop_w * scale_w > w

        if self.keep_center:
            x1 = int((w-crop_w)/2)
            y1 = int((h-crop_h)/2)
        else:
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

        return scale_w, scale_h, x1, y1, crop_w, crop_h

    def crop_resize(self, img, scale_w, scale_h, x1, y1, crop_w, crop_h):
        img = img[..., y1:y1+crop_h, x1:x1+crop_w]
        # import ipdb;ipdb.set_trace()
        if scale_w != 1 or scale_h != 1:
            img = K.geometry.rescale(img, (scale_h, scale_w), align_corners=True)
        img = img[..., :self.target_h, :self.target_w]
        return img

    # the imgs, flows and disps should be with the same h, w
    # TODO: how to handle the different h w, where in the vo training, the flows and disps are downsampled!
    # TODO: get rid of batchsize, h, w
    def forward(self, batchsize, h, w, imgs = None, flows = None, disps = None): 
        self.target_h = min(self.target_h, h)
        self.target_w = min(self.target_w, w)
        scalelist = []

        # import ipdb;ipdb.set_trace()
        imglist, flowlist, displist = [], [], []
        for k in range(batchsize):
            scale_w, scale_h, x1, y1, crop_w, crop_h = self.generate_random_scale_crop(h, w, self.target_h, self.target_w)

            if imgs is not None:
                img = self.crop_resize(imgs[k], scale_w, scale_h, x1, y1, crop_w, crop_h)
                imglist.append(img)

            if flows is not None:
                flow = self.crop_resize(flows[k], scale_w, scale_h, x1, y1, crop_w, crop_h)
                flow[...,0,:,:] = flow[...,0,:,:] * scale_w
                flow[...,1,:,:] = flow[...,1,:,:] * scale_h
                flowlist.append(flow)

            if disps is not None:
                disp = self.crop_resize(disps[k], scale_w, scale_h, x1, y1, crop_w, crop_h)
                disp = disp * scale_w
                displist.append(disp)

            scalelist.append([scale_h, scale_w])

        return imglist, flowlist, displist, scalelist

        # if imgs is not None:
        #     imgs = torch.stack(imglist, 0)
        # if flows is not None:
        #     flows = torch.stack(flowlist, 0)
        # if disps is not None:
        #     disps = torch.stack(displist, 0)

        # return imgs, flows, disps, scalelist

class RandomHSV(nn.Module):
    '''
    Expecting input size: B x Seq x C x H x W, or B x C x H x W
                          or [Seq x C x H x W] of length B
                          or [C x H x W] of length B
    The resultant output is a few lists, which can be stacked together at the end of the data augmentation
    '''
    def __init__(self, lr_rand = 0.0, prob=0.5):
        super(RandomHSV, self).__init__()
        self.bright_range = [-0.3, 0.3]
        self.contrast_range = [0.6, 2.0]
        self.hue_range = [-3.14, 3.14]
        self.satuation_range = [0.0, 3.0]

        self.lr_rand = lr_rand
        self.prob = prob

    def rand_value(self):
        # h = random.uniform(self.hue_range[0], self.hue_range[1])
        h = max(min(random.gauss(0,0.1),1),-1) * self.hue_range[1]
        s = random.uniform(self.satuation_range[0], self.satuation_range[1])
        v = random.uniform(self.bright_range[0], self.bright_range[1])
        c = random.uniform(self.contrast_range[0], self.contrast_range[1])

        # add variation for the left-right images
        hh = random.gauss(0,0.1) * self.lr_rand * (self.hue_range[1]-self.hue_range[0]) + h
        ss = random.uniform(-1, 1) * self.lr_rand * (self.satuation_range[1]-self.satuation_range[0]) + s
        vv = random.uniform(-1, 1) * self.lr_rand * (self.bright_range[1]-self.bright_range[0]) + v
        cc = random.uniform(-1, 1) * self.lr_rand * (self.contrast_range[1]-self.contrast_range[0]) + c
        
        return h,s,v,c,hh,ss,vv,cc


    def forward(self, batchsize, img0 = None, img1 = None):
        img0list, img1list = [], []
        for k in range(batchsize):
            adjustbright    = True if random.random()>self.prob else False
            adjustcontrast  = True if random.random()>self.prob else False
            adjusthue       = True if random.random()>self.prob else False
            adjustsatuation = True if random.random()>self.prob else False
            h,s,v,c,hh,ss,vv,cc = self.rand_value()
            # print(h,s,v,c,hh,ss,vv,cc)
            if img0 is not None:
                img = img0[k]
                img = K.enhance.adjust_brightness(img, v) if adjustbright else img
                img = K.enhance.adjust_contrast(img, c) if adjustcontrast else img
                img = K.enhance.adjust_hue(img, h) if adjusthue else img
                img = K.enhance.adjust_saturation(img, s) if adjustsatuation else img
                img0list.append(img)
            if img1 is not None:
                img = img1[k]
                img = K.enhance.adjust_brightness(img, vv) if adjustbright else img
                img = K.enhance.adjust_contrast(img, cc) if adjustcontrast else img
                img = K.enhance.adjust_hue(img, hh) if adjusthue else img
                img = K.enhance.adjust_saturation(img, ss) if adjustsatuation else img
                img1list.append(img)

        return img0list, img1list

        # if img0 is not None:
        #     img0 = torch.stack(img0list, 0)
        # if img1 is not None:
        #     img1 = torch.stack(img1list, 0)

        # return img0, img1

class RandomFlip(nn.Module):
    '''
    For training flow and stereo
    Expecting input size: B x Seq x C x H x W, or B x C x H x W
                          or [Seq x C x H x W] of length B
                          or [C x H x W] of length B
    The resultant output is a few lists, which can be stacked together at the end of the data augmentation
    '''
    def __init__(self, ud, lr, prob=0.5):
        super(RandomFlip, self).__init__()
        self.ud = ud
        self.lr = lr
        self.prob = prob

    def forward(self, batch, imgs, flows = None, disps = None):
        imglist, flowlist, displist = [], [], []
        if disps is not None:
            assert self.lr==False, "Cannot flip lr in stereo task!"
        for k in range(batch):
            flipud = True if (random.random()>self.prob and self.ud) else False
            fliplr = True if (random.random()>self.prob and self.lr) else False

            img = imgs[k]
            if flipud:
                img = K.geometry.transform.vflip(img)
            if fliplr:
                img = K.geometry.transform.hflip(img)
            imglist.append(img)

            if flows is not None:
                flow = flows[k]
                if flipud:
                    flow = K.geometry.transform.vflip(flow)
                    flow[...,1,:,:] = -flow[...,1,:,:]
                if fliplr:
                    flow = K.geometry.transform.hflip(flow)
                    flow[...,0,:,:] = -flow[...,0,:,:]
                flowlist.append(flow)

            if disps is not None:
                disp = disps[k]
                if flipud:
                    disp = K.geometry.transform.vflip(disp)
                displist.append(disp)

        return imglist, flowlist, displist

class Normalize(nn.Module):
    '''
    Expecting input size: B x Seq x C x H x W, or B x C x H x W, or ... x C x H x W
    This should be called after all the batches are stacked
    '''
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def forward(self, img0, img1=None):
        img0shape = img0.shape
        if img0.ndim > 4:
            newshape = (-1,) + img0shape[-3:]
            img0 = img0.view(newshape)
        img0 = K.enhance.normalize(img0, self.mean, self.std)
        img0 = img0.view(img0shape)

        if img1 is not None:
            img1shape = img1.shape
            if img1.ndim > 4:
                newshape = (-1,) + img1shape[-3:]
                img1 = img1.view(newshape)
            img1 = K.enhance.normalize(img1, self.mean, self.std)
            img1 = img1.view(img1shape) 

            return img0, img1

        return img0

# Already covered by RCR
# class CropCenter(nn.Module):
#     def __init__(self, ):
#         super(CropCenter, self).__init__()

#     def forward(self, img):
#         pass

# class ResizeCrop(nn.Module):
#     def __init__(self, size, resize_scale=1.0):
#         super(ResizeCrop, self).__init__()

#     def forward(self, img):
#         pass

class RandomRotate(nn.Module):
    def __init__(self, maxangle=1.0, sameforseq=False, prob=0.5):
        '''
        do same rotation for data in one seq, if the data is in sequential form
        '''
        super(RandomRotate, self).__init__()
        self.maxangle = maxangle
        self.sameforseq = sameforseq
        self.prob = prob

    def get_rotate_params(self, batchsize, seqnum, h, w):
        anglelist = []
        centerlist = []
        for k in range(batchsize):
            if random.random() < self.prob: # add rotation for this batch
                if self.sameforseq:
                    anglelist.extend([(random.random() *2 -1) * self.maxangle] * seqnum)
                    centerlist.extend([[random.randint(0, h-1), random.randint(0, w-1)] ] * seqnum)
                else:
                    anglelist.extend([(random.random() *2 -1)*self.maxangle for k in range(seqnum)])
                    centerlist.extend([[random.randint(0, h-1), random.randint(0, w-1)] for k in range(seqnum)])
            else:
                anglelist.extend([0.0,] * seqnum)
                centerlist.extend([[0,0] ] * seqnum)
        rotate_center = torch.tensor(centerlist, dtype=torch.float32)
        rotate_angle = torch.tensor(anglelist, dtype=torch.float32)
        return rotate_angle, rotate_center

    # def rotate_image(self, image):
    #     imgnum = image.shape[0]
    #     if self.sameforseq:
    #         angle = [(random.random() *2 -1) * self.maxangle]* imgnum
    #         center = [[random.randint(0, image.shape[1]-1), random.randint(0, image.shape[2]-1)] ] * imgnum
    #     else:
    #         angle = [(random.random() *2 -1)*self.maxangle for k in range(imgnum)]
    #         center = [[random.randint(0, image.shape[1]-1), random.randint(0, image.shape[2]-1)] for k in range(imgnum)]
    #     rotate_center = torch.tensor(center, dtype=torch.float32).to(image.device)
    #     rotate_angle = torch.tensor(angle, dtype=torch.float32).to(image.device)
    #     image = K.geometry.transform.rotate(image, rotate_angle, center=rotate_center)
    #     return image
    
    def forward(self, img1):
        '''
        Expecting input size: B x Seq x C x H x W, or B x C x H x W
                            or [Seq x C x H x W] of length B
                            or [C x H x W] of length B
        The resultant output is a few lists, which can be stacked together at the end of the data augmentation
        '''
        # import ipdb;ipdb.set_trace()
        # batchsize = get_batch_size(img1)
        # img1list = []
        # for k in range(batchsize):
        #     img = img1[k]
        #     if random.random() > 0.5:
        #         imgshape = img.shape
        #         newshape = (-1,) + imgshape[-3:] # view to 1 x c x h x w
        #         img = img.view(newshape)
        #         img = self.rotate_image(img)
        #         img = img.view(imgshape)
        #     img1list.append(img)

        # return img1list
        if img1.ndim == 4:
            batch, c, h, w = img1.shape
            seq = 1
        elif img1.ndim == 5:
            batch, seq, c, h, w = img1.shape
        else:
            assert False, "RandomRotate: Unsupport shape {}".format(img1.shape)

        rotate_angle, rotate_center = self.get_rotate_params(batch, seq, h, w)
        img1shape = img1.shape
        newshape = (-1, ) + img1shape[-3:]
        img1 = img1.view(newshape)
        rotate_angle, rotate_center = rotate_angle.to(img1.device), rotate_center.to(img1.device)
        img1 = K.geometry.transform.rotate(img1, rotate_angle, center=rotate_center)
        img1 = img1.view(img1shape)

        return img1

class RandomUncertainty(nn.Module):
    '''
    Generate random patches of noisy region and corresponding uncertainty
    '''
    def __init__(self, patchnum=5, data_max_bias=1.0, data_noise_scale=0.5): #NoiseSigma, NoiseMean
        '''
        data_max_bias: max bias value add to data
        data_noise_scale: the scale of white noise added to data
        mask_noise_scale: the scale of white noise add
        '''
        super(RandomUncertainty, self).__init__()
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
        bias_base = torch.rand(c)*2-1 # (-1, 1) x c
        bias = bias_base * self.data_max_bias # (-b, b) x c
        whitenoise_base = torch.randn(c, rh, rw) # (-1, 1)
        whitenoise = whitenoise_base * self.data_noise_scale # (-n, n)
        data_noise = bias.view(c,1,1) + whitenoise 
        # generate mask according to the data noise
        mask_whitenoise = torch.abs(torch.randn(rh, rw)) # (0, ~2)
        mask = torch.clip(mask_whitenoise-1, 0, 1 ) # (0, m * mask_base)

        return data_noise, mask

    def random_mask_flow(self, img, mask, maxratio=0.5, maxholes=5):
        '''
        img: c x h x w, c could be 1
        '''
        # import ipdb;ipdb.set_trace()
        c, h, w = img.shape
        # mask = np.random.rand(h,w) * 2 - 4 # generate the background ranging from (-4, -2)
        img_noise = torch.clone(img)

        sx, sy, rh, rw = self.random_patch(w, h, maxratio, minratio=0.1)
        data_noise, mask_noise = self.generate_noise(rh, rw, c)
        data_noise, mask_noise = data_noise.to(img.device), mask_noise.to(img.device)

        # add a few holes 
        for k in range(random.randint(0, maxholes)): 
            hx, hy, hh, hw = self.random_patch(rw, rh, maxratio, minratio=0.1)
            # hx, hy = sx + hx, sy + hy
            data_noise[:, hx:hx+hh, hy:hy+hw] = 0
            mask_noise[hx:hx+hh, hy:hy+hw] = torch.clip(1 - torch.randn(hh,hw) * 0.02, 0, 1)

        img_noise[:,sx:sx+rh, sy:sy+rw] = img_noise[:,sx:sx+rh, sy:sy+rw] + data_noise
        mask[sx:sx+rh, sy:sy+rw] = mask_noise

        return img_noise, mask

    def random_mask_depth(self, img, mask, maxratio=0.5, maxholes=5):
        '''
        img: h x w x c, c could be 1
        '''
        # import ipdb;ipdb.set_trace()
        c, h, w = img.shape
        # mask = np.random.rand(h,w) * 2 - 4 # generate the background ranging from (-4, -2)
        img_noise = img.clone()
        imgmin = img.min()
        sx, sy, rh, rw = self.random_patch(w, h, maxratio, minratio=0.1)
        data_noise, mask_noise = self.generate_noise(rh, rw, c)
        img_noise[:,sx:sx+rh, sy:sy+rw] = img_noise[:,sx:sx+rh, sy:sy+rw] + data_noise
        img_noise = torch.clip(img_noise, imgmin, None)
        
        mask[sx:sx+rh, sy:sy+rw] = mask_noise

        # add a few holes 
        for k in range(random.randint(0, maxholes)): 
            hx, hy, hh, hw = self.random_patch(rw, rh, maxratio, minratio=0.1)
            hx, hy = sx + hx, sy + hy
            img_noise[hx:hx+hh, hy:hy+hw] = img[hx:hx+hh, hy:hy+hw]
            mask[hx:hx+hh, hy:hy+hw] = torch.clip(1 - torch.randn(hh,hw) * 0.02, 0, 1)

        return img_noise, mask

    def forward(self, flows = None, depths = None ):
        '''
        Each flow in a seq has different unc  
        flow: b x seq x c x h x w or b x c x h x w, flow after normalization
        depth: b x seq x c x h x w or b x c x h x w, 
        return: flow, flow_unc, depth, depth_unc w/ corresponding size
        '''

        if flows is not None:
            flowshape = flows.shape
            flows = flows.view((-1,)+flowshape[-3:]) # N x c x h x w
            dataseq = []
            uncseq = []
            for k in range(flows.shape[0]):
                flow = flows[k]
                c, h, w = flow.shape
                unc_mask = torch.clip(1 - torch.randn(h,w) * 0.02, 0, 1).to(flows.device)  # generate the certain background (0.98, 1)
                for k in range(random.randint(0,self.patchnum)):
                    flow, unc_mask = self.random_mask_flow(flow, unc_mask)
                dataseq.append(flow)
                uncseq.append(unc_mask)
            
            # import ipdb;ipdb.set_trace()
            dataflows = torch.stack(dataseq, 0).view(flowshape)
            maskshape = flowshape[:-3]+(1,)+flowshape[3:]
            dataflowunc = torch.stack(uncseq, 0).view(maskshape)
            flows = torch.cat((dataflows, dataflowunc), dim=-3)

        if depths is not None:
            depthshape = depth.shape
            depths = depths.view((-1,)+depthshape[-3:])
            dataseq = []
            uncseq = []
            for k in range(len(depths.shape[0])):
                depth = depths[k]
                c, h, w = depth.shape
                unc_mask = torch.clip(1 - torch.randn(h,w) * 0.02, 0, 1).to(flows.device)  # generate the certain background (0.98, 1)
                # depth = depth.reshape(h,w,1)
                for k in range(random.randint(0,self.patchnum)):
                    depth, unc_mask = self.random_mask_depth(depth, unc_mask)
                dataseq.append(depth.squeeze(-1)).view(depthshape)
                uncseq.append(unc_mask).view(depthshape)
            datadepths = torch.stack(dataseq, 0).view(depthshape)
            datadepthunc = torch.stack(uncseq, 0).view(depthshape)
            depths = torch.cat((datadepths, datadepthunc), dim=-3)

        return flows, depths

class FlowDataTransform(nn.Module):
    def __init__(self, input_size, data_augment=False, resize_factor=1.0, rand_hsv=0.0, flow_norm_factor=1.0):
        super(FlowDataTransform, self).__init__()

        self.data_augment = data_augment
        if data_augment:
            self.rcr = RCR(input_size, max_scale=resize_factor)
            self.hsv = RandomHSV(lr_rand = rand_hsv)
            self.flip = RandomFlip(ud=True, lr=True)
        else:
            self.rc = RCR(input_size, keep_center=True, fix_scale=(resize_factor, resize_factor))
        self.norm = Normalize()

        self.flow_norm_factor = flow_norm_factor

    def forward(self, img, flow):
        batchsize = img.shape[0]
        h, w = img.shape[-2], img.shape[-1]
        img = img/255.0
        # import ipdb;ipdb.set_trace()
        if self.data_augment:
            imglist, flowlist, _, _ = self.rcr(batchsize, h, w, imgs = img, flows = flow)
            imglist, _ = self.hsv(batchsize, img0 = imglist)
            imglist, flowlist, _ = self.flip(batchsize, imgs = imglist, flows = flowlist)
        else:
            imglist, flowlist, _, _ = self.rc(batchsize, h, w, imgs = img, flows = flow)

        img = torch.stack(imglist, 0)
        flow = torch.stack(flowlist, 0)
        img = self.norm(img)
        if self.flow_norm_factor != 1:
            flow = flow * self.flow_norm_factor

        return img, flow

class StereoDataTransform(nn.Module):
    def __init__(self, input_size, data_augment=False, resize_factor=1.0, rand_hsv=0.0, random_rotate_rightimg=0, stereo_norm_factor=1.0):
        super(StereoDataTransform, self).__init__()
        self.data_augment = data_augment
        if data_augment:
            self.rcr = RCR(input_size, max_scale=resize_factor)
            self.hsv = RandomHSV(lr_rand = rand_hsv)
            self.flip = RandomFlip(ud=True, lr=False)
            if random_rotate_rightimg>0:
                self.rotate = RandomRotate(maxangle=random_rotate_rightimg)
        else:
            self.rc = RCR(input_size, keep_center=True, fix_scale=(resize_factor, resize_factor))
        self.norm = Normalize()

        self.random_rotate_rightimg = random_rotate_rightimg
        self.stereo_norm_factor = stereo_norm_factor

    def forward(self, img0, img1, disp):
        '''
        img0: b x c x h x w
        img1: b x c x h x w
        disp: b x c x h x w
        return: b x c x h x w
        '''
        batchsize = img0.shape[0]
        h, w = img0.shape[-2], img0.shape[-1]
        # import ipdb;ipdb.set_trace()
        if self.data_augment:
            img0 = img0/255.0
            img1 = img1/255.0
            img0list, img1list = self.hsv(batchsize, img0 = img0, img1 = img1)
            imglist = [torch.cat((img0, img1), dim=0) for img0,img1 in zip(img0list, img1list)] # [2c x h x w] x b
            imglist, _, displist, _ = self.rcr(batchsize, h, w, imgs = imglist, disps = disp)
            imglist, _, displist = self.flip(batchsize, imgs = imglist, disps = displist)
        else:
            img = torch.cat([img0, img1], dim=1) # b x 2c x h x w
            img = img/255.0
            imglist, _, displist, _ = self.rc(batchsize, h, w, imgs = img, disps = disp)
        # imglist: [2c x h x w]
        # import ipdb;ipdb.set_trace()
        img0list = [img[:3] for img in imglist]
        img1list = [img[3:] for img in imglist]
        img0 = torch.stack(img0list, 0)
        img1 = torch.stack(img1list, 0)
        if self.data_augment and self.random_rotate_rightimg > 0:
            img1 = self.rotate(img1)        
        disp = torch.stack(displist, 0)

        img0 = self.norm(img0)
        img1 = self.norm(img1)
        if self.stereo_norm_factor != 1:
            disp = disp * self.stereo_norm_factor

        return img0, img1, disp

class FlowVODataTransform(nn.Module):
    def __init__(self, input_size, resize_factor=1.0, flow_norm_factor=1.0, uncertainty = False):
        super(FlowVODataTransform, self).__init__()

        if resize_factor > 1.0:
            self.rcr = RCR(input_size, max_scale=resize_factor)
        else:
            self.rc = RCR(input_size, keep_center=True, fix_scale=(resize_factor, resize_factor))

        if uncertainty:
            self.randunc =  RandomUncertainty( data_max_bias=1.0, data_noise_scale=0.5)

        self.resize_factor = resize_factor
        self.flow_norm_factor = flow_norm_factor
        self.uncertainty = uncertainty

    def forward(self, flow, intrinsics):
        '''
        flow: b x 2 x h x w
        intrinsics: b x 2 x h x w
        return: b x 4 x h x w
        '''
        batchsize = flow.shape[0]
        h, w = flow.shape[-2], flow.shape[-1]
        # import ipdb;ipdb.set_trace()
        if self.resize_factor > 1.0:
            intrinsicslist, flowlist, _, _ = self.rcr(batchsize, h, w, imgs = intrinsics, flows = flow)
        else:
            intrinsicslist, flowlist, _, _ = self.rc(batchsize, h, w, imgs = intrinsics, flows = flow)

        flow = torch.stack(flowlist, 0)
        intrinsics = torch.stack(intrinsicslist, 0)

        if self.flow_norm_factor != 1:
            flow = flow * self.flow_norm_factor

        if self.uncertainty:
            flow_unc,_ = self.randunc(flow)
            flow = torch.cat((flow, flow_unc), dim=-3)
            
        return flow, intrinsics

