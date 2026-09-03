from torch.utils.data import DataLoader
import numpy as np
from os.path import isfile
import time
import numbers

from .data_roots import *
from .DataSplitter import DataSplitter
from .DataCacher import DataCacher
from .RAMDataset import RAMDataset
from .DatasetConfigParser import DatasetConfigParser

def parse_inputfile( inputfile):
    '''
    trajlist: [TRAJ0, TRAJ1, ...]
    trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
    framelist: [[FRAMESTR0, FRAMESTR1, ...],[FRAMESTR_K, FRAMESTR_K+1, ...], ...]
    '''
    with open(inputfile,'r') as f:
        lines = f.readlines()
    trajlist, trajlenlist, framelist = [], [], []
    ind = 0
    while ind<len(lines):
        line = lines[ind].strip()
        traj, trajlen = line.split(' ')
        trajlen = int(trajlen)
        trajlist.append(traj)
        trajlenlist.append(trajlen)
        ind += 1
        frames = []
        for k in range(trajlen):
            if ind>=len(lines):
                print("Datafile Error: {}, line {}...".format(inputfile, ind))
                raise Exception("Datafile Error: {}, line {}...".format(inputfile, ind))
            line = lines[ind].strip()
            frames.append(line)
            ind += 1
        framelist.append(frames)
    totalframenum = sum(trajlenlist)
    print('Read {} trajectories, including {} frames'.format(len(trajlist), totalframenum))
    return trajlist, trajlenlist, framelist, totalframenum

def parse_datatype(datatype):

    if datatype == 'tartan-c' or datatype == 'tartan-z':
        from .CacherTartanAirDataset import CacherTartanAirDataset
        return CacherTartanAirDataset
    if datatype == 'tartan':
        from .CacherTartanAirDataset import CacherTartanAirDatasetNoCompress
        return CacherTartanAirDatasetNoCompress

    if datatype.startswith('sintel'):
        from .CacherFlowDataset import CacherSintelDataset
        return CacherSintelDataset
    if datatype.startswith('chairs'):
        from .CacherFlowDataset import CacherChairsDataset
        return CacherChairsDataset
    if datatype.startswith('flying'):
        from .CacherFlowDataset import CacherFlyingDataset
        return CacherFlyingDataset
    # elif datatype == 'kitti': # TODO
    #     DataSetType = KITTIFlowDataset
        # lossmask = True

    if datatype == 'sceneflow':
        from .CacherStereoDataset import CacherSceneflowDataset
        return CacherSceneflowDataset
    if datatype == 'kitti-stereo':
        from .CacherStereoDataset import CacherKittiDataset
        return CacherKittiDataset
    # if datatype == 'euroc':
    #     from Datasets.eurocDataset import EurocDataset
    #     return EurocDataset

    # TODO: update the following datasets            
    if datatype == 'euroc':
        from Datasets.eurocDataset import EurocDataset
        return EurocDataset
    if datatype == 'kitti-vo':
        from Datasets.kittiDataset import KittiVODataset
        return KittiVODataset

    print ('MultiDatasets: Unknow train datatype {}!!'.format(datatype))
    assert False

# TODO: fill all the remaining transforms
def parse_transform(task):
    if task == 'flow':
        from .augmentation import FlowDataTransform
        return FlowDataTransform
    if task == 'stereo':
        from .augmentation import StereoDataTransform
        return StereoDataTransform
    if task == 'flowvo':
        from .augmentation import FlowVODataTransform
        return FlowVODataTransform
    if task == 'monovo':
        pass
    if task == 'stereo-flowvo':
        pass
    if task == 'stereovo':
        pass

def parse_datasize(modalities, cacher_param, frame_skip, random_blur):
    '''
    output modalities_sizes for the data cacher
    handle flow skip
    handle blur image 
    '''
    loadsizelist = cacher_param['load_size']
    cacher_modalities = modalities.copy()
    for k in range(len(cacher_modalities)): # convert flow to flow2/flow4/flow6 if skip sampling
        if cacher_modalities[k] == 'flow':
            if frame_skip>0:
                cacher_modalities[k] = 'flow'+str(frame_skip+1)
        if cacher_modalities[k] == 'img0': # only img0 has blury version for now
            if random_blur > 0:
                cacher_modalities.append('img0blur')
                if not isinstance(loadsizelist[0] , numbers.Number):
                    loadsizelist.append(loadsizelist[k])
    if isinstance(loadsizelist[0] , numbers.Number): # only one size is specified, use it for all the modalities
        modalities_sizes = {mod: loadsizelist for mod in cacher_modalities if (mod not in {'motion', 'imu'}) } 
    else:
        assert len(modalities) == len(loadsizelist), "MultiDatasets: modality number {} should equal to load_size number {}".format(len(modalities), len(loadsizelist))
        modalities_sizes = {mod: ss for (mod, ss) in zip(cacher_modalities, loadsizelist)}
    return modalities_sizes

class MultiDatasets(object):

    def __init__(self, dataset_specfile, 
                       platform, 
                       batch, workernum, 
                       shuffle=True):
        assert isfile(dataset_specfile), "MultiDatasetsBase: Cannot find spec file {}".format(dataset_specfile)
        configparser = DatasetConfigParser()
        dataconfigs = configparser.parse_from_fp(dataset_specfile)
        modalities = dataconfigs['modalities']
        input_seqlen = dataconfigs['input_seqlen']
        assert len(modalities)==len(input_seqlen), "MultiDatasets: modality number {} should equal to input_seqlen number {}".format(len(modalities), len(input_seqlen))
        self.modalities_lengths = {mod: ll for (mod, ll) in zip(modalities, input_seqlen) } 
        self.datasetNum = len(dataconfigs['data'])

        # self.numDataset = len(dataconfigs)
        # self.loss_mask = [False] * self.numDataset
        self.platform = platform
        self.batch = batch
        self.workernum = workernum
        self.shuffle = shuffle

        self.datacachers = [ ] 
        self.datasets = [None, ] * self.datasetNum
        self.dataloaders = [None, ] * self.datasetNum
        self.dataiters = [None, ] * self.datasetNum
        # self.lossmasks = []
        self.transforms = []
        self.datalens = []
        self.datasetparams = []

        self.init_datasets(dataconfigs)

    def init_datasets(self, dataconfigs):
        task = dataconfigs['task']
        modalities = dataconfigs['modalities']
        for datafile, params in dataconfigs['data'].items():
            trajlist, trajlenlist, framelist, framenum = parse_inputfile(datafile)
            subsetframenum = params['cacher']['subset_framenum']
            data_splitter = DataSplitter(trajlist, trajlenlist, framelist, subsetframenum, shuffle=True) 
            self.datalens.append(subsetframenum)

            # parse load_size
            cacher_param = params['cacher']
            frame_skip = params['dataset']['frame_skip'] if 'frame_skip' in params['dataset'] else 0
            random_blur = params['dataset']['random_blur'] if 'random_blur' in params['dataset'] else 0
            modalities_sizes = parse_datasize(modalities, cacher_param, frame_skip, random_blur)
            
            # create a DataCacher
            datatype = cacher_param['datatype']
            workernum = cacher_param['worker_num']
            data_root = DataRoot[self.platform][datatype]
            DataType = parse_datatype(datatype)
            datacacher = DataCacher(modalities_sizes, DataType, data_root, data_splitter, num_worker=workernum, batch_size=1)
            self.datacachers.append(datacacher)

            # create transform
            TransfromClass = parse_transform(task)
            transform_param = params['transform']
            transformclass = TransfromClass(**transform_param)
            # transform = transformclass.get_transformlist() 
            self.transforms.append(transformclass)

            # parameters for the RAMDataset
            dataset_param = params['dataset']
            self.datasetparams.append(dataset_param)

        self.accDataLens = np.cumsum(self.datalens).astype(np.float64)/np.sum(self.datalens)    

        # wait for all datacacher being ready
        for k, datacacher in enumerate(self.datacachers):
            while not datacacher.new_buffer_available:
                time.sleep(1)
            self.datacachers[k].switch_buffer()
            dataset = RAMDataset(self.datacachers[k], \
                                self.modalities_lengths, \
                                **self.datasetparams[k], \
                                transform = None ) #self.transforms[k])
            dataloader = DataLoader(dataset, batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            self.datasets[k] = dataset
            self.dataloaders[k] = dataloader
            self.dataiters[k] = iter(dataloader)

    def load_sample(self, fullbatch=True):
        # Randomly pick the dataset in the list
        randnum = np.random.rand()
        datasetInd = 0 
        while randnum > self.accDataLens[datasetInd]: # 
            datasetInd += 1

        # load sample from the dataloader
        try:
            sample = self.dataiters[datasetInd].next()
            if sample[list(sample.keys())[0]].shape[0] < self.batch and (fullbatch is True): # the imcomplete batch is thrown away
                # self.datasets[datasetInd].set_epoch_complete()
                # self.dataiters[datasetInd] = iter(self.dataloaders[datasetInd])
                # sample = self.dataiters[datasetInd].next()
                sample = self.dataiters[datasetInd].next()
        except StopIteration:
            # self.datasets[datasetInd].set_epoch_complete()
            if self.datacachers[datasetInd].new_buffer_available : 
                self.datacachers[datasetInd].switch_buffer()
                self.datasets[datasetInd] = RAMDataset(self.datacachers[datasetInd], \
                                    self.modalities_lengths, \
                                    **self.datasetparams[datasetInd], \
                                    transform = None) #self.transforms[datasetInd])
                self.dataloaders[datasetInd] = DataLoader(self.datasets[datasetInd], batch_size=self.batch, shuffle=self.shuffle, num_workers=self.workernum)
            self.dataiters[datasetInd] = iter(self.dataloaders[datasetInd])
            sample = self.dataiters[datasetInd].next()
        # print("sample time: {}".format(time.time()-cachertime))
        return sample, self.transforms[datasetInd]

    def stop_cachers(self):
        for datacacher in self.datacachers:
            datacacher.stop_cache()

if __name__ == '__main__':

    def vis_intrinsics(intrinsics):
        dispintrinsics = intrinsics.cpu().numpy().transpose(1,2,0) 
        dispintrinsics = np.clip(dispintrinsics * 255, 0, 255).astype(np.uint8)
        dispintrinsics = np.concatenate((dispintrinsics, np.zeros((intrinsics.shape[1],intrinsics.shape[2],1),dtype=np.uint8)), axis=2)
        return dispintrinsics

    # ===== Test MultiDatasets ======
    # from .utils import visflow, tensor2img
    import time
    from .DatasetConfigParser import DatasetConfigParser
    from .utils import visflow
    import cv2
    # dataset_specfile = '/data/datasets/wenshanw/workspace/geometry_vision/Datacacher/dataspec/flow_train_debug_cluster.yaml'
    # dataset_specfile = '/home/amigo/workspace/pytorch/geometry_vision/Datacacher/dataspec/flow_train_chairs_local.yaml'
    dataset_specfile = '/home/amigo/workspace/pytorch/geometry_vision/Datacacher/dataspec/flowvo_train_local.yaml'
    # configparser = DatasetConfigParser()
    # dataconfigs = configparser.parse_from_fp(dataset_specfile)
    batch = 100
    trainDataloader = MultiDatasets(dataset_specfile, 
                       'local', 
                       batch=batch, workernum=6, 
                       shuffle=True)
    tic = time.time()
    num = 100                       
    for k in range(num):
        sample, transform = trainDataloader.load_sample()
        print(sample.keys(), sample['intrinsics'].shape, sample['flow'].shape)
        flow, intrinsics = transform(sample['flow'].squeeze(1), sample['intrinsics'])
        # time.sleep(0.02)
        import ipdb;ipdb.set_trace()
        for b in range(batch):
            # import ipdb;ipdb.set_trace()
            dispflow0 = visflow(sample['flow'][b,0].numpy().transpose(1,2,0))
            dispintrinsics0 = vis_intrinsics(sample['intrinsics'][b])
            disp0 = np.concatenate((dispflow0, dispintrinsics0), axis=0)
            flowk = flow[b][0] / 0.05
            intrinsicsk = intrinsics[b]
            dispflow = visflow(flowk.cpu().numpy().transpose(1,2,0))
            dispintrinsics = vis_intrinsics(intrinsicsk)
            disp1 = np.concatenate((dispflow, dispintrinsics), axis=0)
            dispdisp = np.concatenate((disp0, disp1), axis=0)
            dispdisp = cv2.resize(dispdisp, (0,0), fx=2.0, fy=2.0)
            cv2.imshow('img', dispdisp)
            cv2.waitKey(0)            
    print((time.time()-tic))
    trainDataloader.stop_cachers()
