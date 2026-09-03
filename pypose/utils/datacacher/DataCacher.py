import torch
from torch.utils.data import DataLoader
import time

import threading
torch.multiprocessing.set_sharing_strategy('file_system')

from .TrajBuffer import TrajBuffer


class DataCacher(object):

    def __init__(self, modalities_sizes, cacher_dataset, data_root, data_splitter, num_worker, batch_size=1):
        '''
        modalities_sizes: { 'img0':(h,w),
                            'img1':(h,w),
                            'img0blur':(h,w),
                            'depth0':(h,w),
                            'depth1':(h,w),
                            'flow':(h,w),
                            'fmask':(h,w),
                            'flow2':(h,w),
                            'fmask2':(h,w),
                            'flow4':(h,w),
                            'fmask4':(h,w),
                            'flow6':(h,w),
                            'fmask6':(h,w)}
        cacher_dataset: the pytorch dataloader that loads data from hard drive to RAM 
        data_root: the root directory of the dataset
        num_worker: the number of workers
        batch_size: the batch size, 1 is best as tested on my local machine
        The sizes defined in the modalities_sizes are are the sizes required for training, 
        if the loaded data is in different shape with what defined in modalities_sizes, the data will be resized
        '''
        self.modalities_sizes = modalities_sizes
        self.datatypes = list(self.modalities_sizes.keys())
        self.num_worker = num_worker
        self.batch_size = batch_size
        self.cacher_dataset = cacher_dataset
        self.data_root = data_root
        self.data_splitter = data_splitter # split the whole dataset into subsets

        # initialize two buffers
        self.loading_buffer = None
        self.ready_buffer = None
        self.loading_a = False
        self.loading_b = False
        self.new_buffer_available = False
        # This following lines won't allocate RAM memory yet
        self.buffer_a = TrajBuffer(self.modalities_sizes)
        self.buffer_b = TrajBuffer(self.modalities_sizes)

        # initialize a dataloader
        self.dataiter = None
        self.stop_flag = False
        self.loading_b = False
        self.loading_a = True
        self.loading_buffer = self.buffer_a
        self.reset_buffer() # this will allocate the memory

        # run datacacher in a sperate thread
        th = threading.Thread(target=self.run)
        th.start()

    def reset_buffer(self):
        '''
        This function allocates the shared memory
        '''
        trajlist, trajlenlist, framelist, framenum = self.data_splitter.get_next_split()
        self.loading_buffer.reset(framenum, trajlist, trajlenlist, framelist)

        dataset = self.cacher_dataset(self.modalities_sizes, trajlist, trajlenlist, framelist, self.data_root)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)#, persistent_workers=True)
        self.dataiter = iter(dataloader)
        self.new_buffer_available = False
        self.loading_ind = 0
        self.starttime = time.time()

    def switch_buffer(self):
        if (self.loading_b and self.buffer_b.full):
            # start to load buffer a
            self.loading_b = False
            self.loading_a = True
            self.loading_buffer = self.buffer_a
            self.ready_buffer = self.buffer_b

        elif self.loading_a and self.buffer_a.full:
            # switch to buffer b
            self.loading_a = False
            self.loading_b = True
            self.loading_buffer = self.buffer_b
            self.ready_buffer = self.buffer_a

        else:
            return

        self.reset_buffer()

    def __getitem__(self, index):
        return self.ready_buffer[index]

    def run(self):
        # check which buffer is active
        while not self.stop_flag:
            if not self.new_buffer_available:
                # self.switch_buffer()
                try:
                    sample = self.dataiter.next()
                    # if self.loading_ind >= self.loading_buffer.framenum: # debug
                    #     import ipdb;ipdb.set_trace()
                    self.loading_buffer.insert(self.loading_ind, sample)
                    self.loading_ind += self.batch_size
                except StopIteration:
                    print('Buffer loaded: traj {}, frame {}, time {}'.format(len(self.loading_buffer.trajlist),len(self.loading_buffer), time.time()-self.starttime))
                    self.loading_buffer.full = True
                    self.new_buffer_available = True
            else:
                time.sleep(0.1)

    def stop_cache(self):
        self.stop_flag = True

# python -m Datacacher.DataCacher
if __name__=="__main__":
    from .CacherTartanAirDataset import CacherTartanAirDataset,CacherTartanAirDatasetNoCompress
    from .DataSplitter import DataSplitter
    from .utils import visflow, visdepth
    from .MultiDatasets import parse_inputfile
    import cv2
    import numpy as np

    from torch.utils.data import Dataset, DataLoader
    import os

    class MyDataset(Dataset):
        def __init__(self, buffer):
            self.buffer = buffer

        def __getitem__(self, index):
            # if not self.use_cache:
            #     print('Filling cache for index {}'.format(index))
            #     # Add your loading logic here
            #     self.shared_array[index] = torch.randn(c, h, w)
            x = self.buffer[index]# .copy()
            # print('pid ',os.getpid(),' index ',index)
            # print (self.shared_array.shared_array)
            # for k in range(len(self.shared_array.shared_array[index])):
            #     self.shared_array.shared_array[index][k] = -1
            return x

        def __len__(self):
            return len(self.buffer.ready_buffer)

    inputfile = '/home/amigo/workspace/pytorch/geometry_vision/data/tartan_train_local.txt'
    # inputfile = '/home/amigo/workspace/pytorch/geometry_vision/data/tartan_train.txt'
    subsetframenum = 200
    trajlist, trajlenlist, framelist, framenum = parse_inputfile(inputfile)
    modalities_sizes = {'img1':(480,640), 'flow6':(480,640), 'fmask6':(480,640), 'depth0':(480,640)} #
    data_root = '/home/amigo/tmp/data/tartan'
    data_splitter = DataSplitter(trajlist, trajlenlist, framelist, subsetframenum)
    datacacher = DataCacher(modalities_sizes, CacherTartanAirDatasetNoCompress, data_root, data_splitter, num_worker=0, batch_size=1)
    
    while not datacacher.new_buffer_available:
        print('wait for data loading...')
        time.sleep(1)
    # import ipdb;ipdb.set_trace()
    # datacacher.switch_buffer()

    # while True:
    #     if datacacher.new_buffer_available:
    #         datacacher.switch_buffer()
    #     print(datacacher.loading_a, datacacher.loading_b)
    #     # sample = datacacher[5:10:2]
    #     # import ipdb;ipdb.set_trace()
    #     for k in range(subsetframenum):
    #         sample = datacacher[k]
    #         img = sample["img1"]
    #         flow = sample["flow6"]
    #         flowvis = visflow(flow)
    #         # flowvis = cv2.resize(flowvis, (0,0), fx=4, fy=4)
    #         depth = sample["depth0"]
    #         depthvis = visdepth(depth)
    #         fmask = sample["fmask6"]
    #         fmaskvis = (fmask>0).astype(np.uint8)*255
    #         fmaskvis = np.tile(fmaskvis[:,:,np.newaxis], (1, 1, 3))
    #         # fmaskvis = cv2.resize(fmaskvis, (640, 480))
    #         disp = np.concatenate((img,flowvis,depthvis,fmaskvis), axis=1) # 
    #         # if flow.max()==0:
    #         #     print(k, 'flow zeros')
    #         # if fmask.max()==0:
    #         #     print(k, 'fmask zeros')
    #         cv2.imshow('img',disp)
    #         cv2.waitKey(10)
    #         # print(k, img.shape, flow.shape)

    datacacher.switch_buffer()
    dataset = MyDataset(datacacher)
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=2,
        shuffle=False
    )
    dataiter = iter(loader)

    while True:
        try:
            sample = dataiter.next()
            print(sample.keys())
        except StopIteration:
            if datacacher.new_buffer_available:
                datacacher.switch_buffer()
                dataset = MyDataset(datacacher)
                loader = DataLoader(
                    dataset,
                    batch_size=10,
                    num_workers=2,
                    shuffle=False
                )
            dataiter = iter(loader)
        time.sleep(0.1)

        # print(datacacher.loading_a, datacacher.loading_b)
        # # sample = datacacher[5:10:2]
        # # import ipdb;ipdb.set_trace()
        # for k in range(subsetframenum):
        #     sample = datacacher[k]
        #     img = sample["img1"]
        #     flow = sample["flow6"]
        #     flowvis = visflow(flow)
        #     # flowvis = cv2.resize(flowvis, (0,0), fx=4, fy=4)
        #     depth = sample["depth0"]
        #     depthvis = visdepth(depth)
        #     fmask = sample["fmask6"]
        #     fmaskvis = (fmask>0).astype(np.uint8)*255
        #     fmaskvis = np.tile(fmaskvis[:,:,np.newaxis], (1, 1, 3))
        #     # fmaskvis = cv2.resize(fmaskvis, (640, 480))
        #     disp = np.concatenate((img,flowvis,depthvis,fmaskvis), axis=1) # 
        #     # if flow.max()==0:
        #     #     print(k, 'flow zeros')
        #     # if fmask.max()==0:
        #     #     print(k, 'fmask zeros')
        #     cv2.imshow('img',disp)
        #     cv2.waitKey(10)
        #     # print(k, img.shape, flow.shape)
