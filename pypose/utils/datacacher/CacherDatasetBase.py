import cv2
from torch.utils.data import Dataset
import numpy as np

class CacherDatasetBase(Dataset):
    '''
    Load the data from hard drive to RAM
    This is similar to the original TartanAirDataset, but without considering sequencing, frame skipping, data augmentation, normalization, etc. 
    Only image-like data are supported, including image, depth and flow
    Resize the data if necessary
    Note if the flow or disp is resized, the pixel value won't be changed! 
    '''
    def __init__(self, modalities_sizes, trajlist, trajlenlist, framelist, datarootdir=""):
        '''
        supported datatypes: img0,img0blur,img1,depth0,depth1,disp0,disp1,flow,fmask,flow2,fmask2,flow4,fmask4,flow6,fmask6
        '''
        self.modalities_sizes = modalities_sizes
        self.datatypes = list(self.modalities_sizes.keys())
        self.trajlist = trajlist
        self.trajlenlist = trajlenlist
        self.framelist = framelist
        self.dataroot = datarootdir
        self.trajnum = len(trajlist)

        self.framenum = sum(trajlenlist)
        self.acc_trajlen = [0,] + np.cumsum(trajlenlist).tolist() # [0, num[0], num[0]+num[1], ..]

    def __len__(self):
        return self.framenum

    def idx2traj(self, idx):
        '''
        handle the stride and the skip
        return: 1. the relative dir of trajectory 
                2. the frame string 
                3. is the frame at the end of the current trajectory (for loading flow)
        '''
        # import ipdb;ipdb.set_trace()
        for k in range(self.trajnum):
            if idx < self.acc_trajlen[k+1]:
                break

        remainingframes = idx-self.acc_trajlen[k]
        # frameind = self.acc_trajlen[k] + remainingframes
        framestr = self.framelist[k][remainingframes]
        frameindex_inv = self.trajlenlist[k] - remainingframes # is this the last few frames where there might no flow data exists

        return self.trajlist[k], framestr, frameindex_inv

    def __getitem__(self, idx):
        # load images from the harddrive
        # load one frame of multi-modal data
        trajstr, framestr, ind_inv = self.idx2traj(idx)

        sample = {}
        for datatype in self.datatypes: 
            datafile = self.getDataPath(trajstr, framestr, datatype, ind_inv)
            if datatype == 'img0' or datatype == 'img1' or datatype == 'img0blur':
                data = self.load_image(datafile)
                if data is None:
                    print("!!!READ IMG ERROR {}, {}, {}".format(idx, trajstr, framestr, datafile))
            elif datatype == 'depth0' or datatype == 'depth1':
                data = self.load_depth(datafile)
            elif datatype == 'disp0' or datatype == 'disp1':
                data = self.load_disparity(datafile)

            # the following way of handling flow is not elegant
            # because we want to deal with the fact that flow is shorter than other modalities
            # and fmask is returned together with flow (in the compressed case), so we want to avoid loading twice
            elif datatype[:4] == 'flow': # this includes flow, flow2, flow4, ...
                data, fmask = self.load_flow(datafile)
                fmasktype = datatype.replace('flow', 'fmask') # check if the fmask, fmask2, fmask4, ... is also required
                if fmasktype in self.datatypes:
                    if fmask.shape[:2] != self.modalities_sizes[fmasktype]:
                        fmask = cv2.resize(fmask, (self.modalities_sizes[fmasktype][1], self.modalities_sizes[fmasktype][0]), interpolation=cv2.INTER_NEAREST ) 
                    sample[fmasktype] = fmask
            elif datatype[:5] == 'fmask':
                # fmask with flow will be handled above
                flowtype = datatype.replace('fmask', 'flow')
                if flowtype not in self.datatypes: # this fmask doesn't come with a flow
                    _, data = self.load_flow(datafile) # load the mask
                else: # fmaskx should be handled with flowx
                    continue
            else:
                print('Unknow Datatype {}'.format(datatype))

            # resize the data if needed
            if data.shape[:2] != self.modalities_sizes[datatype]:
                data = cv2.resize(data, (self.modalities_sizes[datatype][1], self.modalities_sizes[datatype][0]), interpolation=cv2.INTER_LINEAR )
            sample[datatype] = data
            # print(datatype, data.shape)

        return sample

    def getDataPath(self, trajstr, framestr, datatype, ind_inv):
        raise NotImplementedError

    def load_flow(self, fn):
        """This function should return 2 objects, flow and mask. """
        raise NotImplementedError

    def load_image(self, fn):
        raise NotImplementedError

    def load_depth(self, fn):
        raise NotImplementedError

    def load_disparity(self, fn):
        raise NotImplementedError
