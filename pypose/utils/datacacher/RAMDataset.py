import cv2
import numpy as np

from torch.utils.data import Dataset
from .utils import make_intrinsics_layer

class RAMDataset(Dataset):
    '''
    datacacher
    modalities_lengths: e.g. {"img0": 2, "img1": 1, "flow": 1, "imu": 10}
    the following modalities are supported: 
    img0(img0blur),img1,disp0,disp1,depth0,depth1,flow,fmask,motion,imu

    imu_freq: only useful when imu modality is queried 
    intrinsics: [w, h, fx, fy, ox, oy], used for generating intrinsics layer. No intrinsics layer added if the value is None
    blxfx: used to convert between depth and disparity

    Note it is different from the CacherDataset, flow/flow2/flow4 are not differetiated here, but controled by frame_skip
    Note only one of the flow/flow2/flow4 can be queried in one Dataset
    Note now the img sequence and the flow sequence are coorelated, which means that you can not ask for a image seq with 0 skipping while querying flow2

    When a sequence of data is required, the code will automatically adjust the length of the dataset, to make sure the every modality exists. 
    The IMU has a higher frequency than the other modalities. The frequency is imu_freq x other_freq. 

    If intrinsics is not None, a intrinsics layer is added to the sample
    The intrinsics layer will be scaled wrt the intrinsics_sclae
    '''
    def __init__(self, \
        datacacher, \
        modalities_lengths, \
        imu_freq = 10, \
        intrinsics = None, intrinsics_scale = 1., \
        blxfx = None, \
        transform = None, \
        frame_skip = 0, \
        seq_stride = 1, \
        random_blur = 0.0):  

        super(RAMDataset, self).__init__()
        self.datacacher = datacacher
        self.modalities_lengths = modalities_lengths
        self.datatypelist = list(self.modalities_lengths.keys())
        self.transform = transform
        self.imu_freq = imu_freq
        self.random_blur = random_blur

        self.intrinsics = intrinsics
        self.intrinsics_scale = intrinsics_scale
        self.blxfx = blxfx

        self.frame_skip = frame_skip # sample not consequtively, skip a few frames within a sequences
        self.seq_stride = seq_stride # sample less sequence, skip a few frames between two sequences 

        # initialize the trajectories and figure out the seqlen
        assert datacacher.ready_buffer.full, "Databuffer in RAM is not ready! "
        self.trajlist = datacacher.ready_buffer.trajlist
        self.trajlenlist = datacacher.ready_buffer.trajlenlist 
        self.framelist = datacacher.ready_buffer.framelist
        self.dataroot = datacacher.data_root

        self.sample_seq_len = self.calc_seq_len(modalities_lengths, imu_freq)
        self.seqnumlist = self.parse_length(self.trajlenlist, frame_skip, seq_stride, self.sample_seq_len)

        self.framenumFromFile = sum(self.trajlenlist)
        self.N = sum(self.seqnumlist)
        self.trajnum = len(self.trajlenlist)
        self.acc_trajlen = [0,] + np.cumsum(self.trajlenlist).tolist()
        self.acc_seqlen = [0,] + np.cumsum(self.seqnumlist).tolist() # [0, num[0], num[0]+num[1], ..]

        if 'motion' in self.modalities_lengths:
            self.motions = self.load_motion()

        if self.intrinsics is not None:
            self.intrinsics_layer = make_intrinsics_layer(self.intrinsics)
            if self.intrinsics_scale != 1:
                self.intrinsics_layer = cv2.resize(self.intrinsics_layer, (0,0), fx=self.intrinsics_scale, fy=self.intrinsics_scale)
            self.intrinsics_layer = self.intrinsics_layer.transpose(2,0,1)

        self.is_epoch_complete = False # flag is set to true after all the data is sampled

        # self.acc_motionlen = [0,] + np.cumsum(self.motionlenlist).tolist() # [0, num[0], num[0]+num[1], ..]
        # self.acc_imulen = [0,] + np.cumsum(self.imulenlist).tolist() # [0, num[0], num[0]+num[1], ..]
        print('Loaded {} sequences from the RAM, which contains {} frames...'.format(self.N, self.framenumFromFile))

    def load_motion(self):
        # import ipdb;ipdb.set_trace()
        print('Loading Motion data from motionfiles...')
        motionlist = []

        for k, trajdir in enumerate(self.trajlist): 
            motiondir = self.dataroot + '/' + trajdir
            motionfile = motiondir + '/motion_left{}.npy'.format(str(self.frame_skip+1) if self.frame_skip>0 else '')
            motions = np.load(motionfile).astype(np.float32)
            # the following code assumes that the starting frame of a trajectory is not from 0
            # and framelist is defined as the frame id that can be converted to int 
            startind = int(self.framelist[k][0])
            endind = int(self.framelist[k][-1])

            motionslen = len(motions)
            trajlen = self.trajlenlist[k]
            # possible cases:
            # startind 0, tranlen 10, motionslen 9, skip 0
                #      0,         10,            8,      1
                #      5,          5,            9,      0
                #      5,          3,            7,      2
            assert startind + trajlen <= motionslen + self.frame_skip + 1, 'Error in load_motion: startind {}, endind {}, traj len {}, motion file len {}'.format(startind, endind, trajlen, motionslen)
            assert endind <= motionslen + self.frame_skip, 'Error in load_motion: endind {}, motion file len {}'.format(endind, motionslen)
            
            # pad the motion at the end of each trajectory
            paddingnum = self.frame_skip + 1
            mmm = motions[startind: startind + trajlen -self.frame_skip -1]
            mmm = np.concatenate((mmm, np.zeros((paddingnum, 6),dtype=np.float32)), axis=0)
            motionlist.append(mmm)

        motions = np.concatenate(motionlist, axis=0)
        print('Loaded {} motion from {} files'.format(len(motions), len(motionlist)))
        
        return motions

    def calc_seq_len(self, modalities_lengths, imu_freq):
        '''
        decide what is the sequence length for cutting the data, considering the different length of different modalities
        For now, all the modalities are at the same frequency except for the IMU which is faster by a factor of 'imu_freq'
        seqlens: the length of seq for each modality
        '''
        maxseqlen = 0
        for ttt, seqlen in modalities_lengths.items():
            if ttt=='imu': # IMU has a higher freqency than other modalities
                seqlen = int((float(seqlen+imu_freq-1)/imu_freq))
            if ttt == 'flow' or ttt == 'fmask' or ttt == 'motion': # if seqlen of flow is equal to or bigger than other modality, add one to the seqlen
                seqlen += 1 # flow and motion need one more frame to calculate the relative change
            if seqlen > maxseqlen:
                maxseqlen = seqlen
        return maxseqlen

    def parse_length(self, trajlenlist, skip, stride, sample_length): 
        '''
        trajlenlist: the length of each trajectory in the dataset
        skip: skip frames within sequence
        stride: skip frames between sequence
        sample_length: the sequence length 
        Return: 
        seqnumlist: the number of sequences in each trajectory
        the length of the whole dataset is the sum of the seqnumlist
        '''
        seqnumlist = []
        # sequence length with skip frame 
        # e.g. x..x..x (sample_length=3, skip=2, seqlen_w_skip=1+(2+1)*(3-1)=7)
        seqlen_w_skip = (skip + 1) * sample_length - skip
        # import ipdb;ipdb.set_trace()
        for trajlen in trajlenlist:
            # x..x..x---------
            # ----x..x..x-----
            # --------x..x..x-
            # ---------x..x..x <== last possible sequence
            #          ^-------> this starting frame number is (trajlen - seqlen_w_skip + 1)
            # stride = 4, skip = 2, sample_length = 3, seqlen_w_skip = 7, trajlen = 16
            # seqnum = (16 - 7)/4 + 1 = 3
            seqnum = int((trajlen - seqlen_w_skip)/ stride) + 1
            if trajlen<seqlen_w_skip:
                seqnum = 0
            seqnumlist.append(seqnum)
        return seqnumlist

    def idx2slice(self, idx):
        '''
        handle the stride and the skip
        return: a slice object for querying the RAM
        '''
        # import ipdb;ipdb.set_trace()
        for k in range(self.trajnum):
            if idx < self.acc_seqlen[k+1]:
                break

        remainingframes = (idx-self.acc_seqlen[k]) * self.seq_stride
        start_frameind = self.acc_trajlen[k] + remainingframes
        end_frameind = start_frameind + (self.frame_skip+1) * (self.sample_seq_len-1)
        assert end_frameind - self.acc_trajlen[k] < self.trajlenlist[k], "Sample a sequence cross two trajectories! This should never happen! "

        slicedict = {}
        for datatype, datalen in self.modalities_lengths.items(): 
            end_frameind = start_frameind + (self.frame_skip+1) * (datalen-1)
            slicedict[datatype] = slice(start_frameind, end_frameind+1, self.frame_skip+1)
        return slicedict 

    def __len__(self):
        return self.N

    def epoch_complete(self):
        return self.is_epoch_complete

    def set_epoch_complete(self):
        self.is_epoch_complete = True

    def __getitem__(self, idx):
        # parse the idx to trajstr
        ramslices = self.idx2slice(idx)

        # sample = self.datacacher[ramslice]
        sample = {}
        # TODO: add motion and imu
        for datatype, datalen in self.modalities_lengths.items(): 
            ramslice = ramslices[datatype]
            if datatype.startswith('img'): # the RAMDataset has consistent types
                sample[datatype] = self.datacacher.ready_buffer.buffer[datatype][ramslice]
                if datatype == 'img0':
                    if np.random.rand() < self.random_blur:
                        sample[datatype] = self.datacacher.ready_buffer.buffer['img0blur'][ramslice]

            elif datatype.startswith('depth'): # depth and disp can be converted
                if datatype in self.datacacher.modalities_sizes:
                    sample[datatype] = self.datacacher.ready_buffer.buffer[datatype][ramslice]
                elif datatype.replace('depth', 'disp') in self.datacacher.modalities_sizes:
                    data = self.datacacher.ready_buffer.buffer[datatype.replace('depth', 'disp')][ramslice]
                    sample[datatype] = self.blxfx/data # convert from disp to depth
                else:
                    assert False, "Cannot load {} from RAM!".format(datatype)

            elif datatype.startswith('disp'): # depth and disp can be converted
                if datatype in self.datacacher.modalities_sizes:
                    sample[datatype] = self.datacacher.ready_buffer.buffer[datatype][ramslice]
                elif datatype.replace('disp', 'depth') in self.datacacher.modalities_sizes:
                    data = self.datacacher.ready_buffer.buffer[datatype.replace('disp', 'depth')][ramslice]
                    sample[datatype] = self.blxfx/data # convert from disp to depth
                else:
                    assert False, "Cannot load {} from RAM!".format(datatype)

            elif datatype == 'flow': # optical flow is different
                ramdatatype = 'flow'+str(self.frame_skip+1) if self.frame_skip>0 else 'flow'
                sample[datatype] = self.datacacher.ready_buffer.buffer[ramdatatype][ramslice]

            elif datatype == 'fmask': 
                ramdatatype = 'fmask'+str(self.frame_skip+1) if self.frame_skip>0 else 'fmask'
                sample[datatype] = self.datacacher.ready_buffer.buffer[ramdatatype][ramslice]

            elif datatype == 'motion':
                motionlist = self.motions[ramslice]
                sample[datatype] = motionlist
            # elif datatype == 'imu': 
            #     imulist = self.load_imu(imuframeind, datalen)
            #     sample[datatype] = imulist
            else:
                print('Unknow Datatype {}'.format(datatype))

        if self.blxfx is not None:
            sample['blxfx'] = self.blxfx # used for convert disp to depth
        if self.intrinsics is not None:
            sample['intrinsics'] = self.intrinsics_layer

        # Transform.
        if ( self.transform is not None):
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    from .CacherTartanAirDataset import CacherTartanAirDataset,CacherTartanAirDatasetNoCompress
    from .DataSplitter import DataSplitter
    from Datasets.utils import visflow, visdepth
    from .DataCacher import DataCacher
    from .MultiDatasets import parse_inputfile
    from torch.utils.data import DataLoader

    import cv2
    import numpy as np
    import time
    inputfile = '/home/amigo/workspace/pytorch/geometry_vision/data/tartan_train_local.txt'
    subsetframenum = 300
    trajlist, trajlenlist, framelist, framenum = parse_inputfile(inputfile)
    modalities_sizes = {'img1':(480,640), 'flow2':(120,160), 'fmask2':(200,300), 'depth0':(480,640)} #
    modalities_lengths = {'img1':3, 'flow':2, 'fmask':2, 'depth0':1, 'disp0':2, 'motion':3} #
    data_root = '/home/amigo/tmp/data/tartan'
    data_splitter = DataSplitter(trajlist, trajlenlist, framelist, subsetframenum)
    datacacher = DataCacher(modalities_sizes, CacherTartanAirDatasetNoCompress, data_root, data_splitter, num_worker=4, batch_size=1)
    while not datacacher.new_buffer_available:
        print('wait for data loading...')
        time.sleep(1)
    # import ipdb;ipdb.set_trace()

    for k in range(100):
        if datacacher.new_buffer_available:
            datacacher.switch_buffer()
            dataset = RAMDataset(datacacher, \
                        modalities_lengths, \
                        imu_freq = 10, \
                        blxfx = 80., \
                        transform = None, \
                        frame_skip = 1, seq_stride = 1)
            dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0)
            dataiter = iter(dataloader)

        sample = dataiter.next()
        print(sample['motion'])
        time.sleep(0.01)
        # print(datacacher.loading_a, datacacher.loading_b)
        # sample = datacacher[5:10:2]
        # import ipdb;ipdb.set_trace()
        # for k in range(subsetframenum):
        #     sample = datacacher[k]
        #     img = sample["img1"]
        #     flow = sample["flow6"]
        #     flowvis = visflow(flow)
        #     flowvis = cv2.resize(flowvis, (0,0), fx=4, fy=4)
        #     depth = sample["depth0"]
        #     depthvis = visdepth(depth)
        #     fmask = sample["fmask6"]
        #     fmaskvis = (fmask>0).astype(np.uint8)*255
        #     fmaskvis = np.tile(fmaskvis[:,:,np.newaxis], (1, 1, 3))
        #     fmaskvis = cv2.resize(fmaskvis, (640, 480))
        #     disp = np.concatenate((img,flowvis,depthvis,fmaskvis), axis=1) # 
        #     if flow.max()==0:
        #         print(k, 'flow zeros')
        #     if fmask.max()==0:
        #         print(k, 'fmask zeros')
        #     cv2.imshow('img',disp)
        #     cv2.waitKey(10)
        #     # print(k, img.shape, flow.shape)
