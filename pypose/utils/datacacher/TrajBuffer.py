from .RAMBuffer import ImageBuffer, DepthBuffer, FlowBuffer, FMaskBuffer

class TrajBuffer(object):
    '''
    Store the multi-modal data in the form of trajectories
    '''
    def __init__(self, modalities_sizes):
        self.modalities_sizes = modalities_sizes
        self.datatypes = list(self.modalities_sizes.keys())
        self.buffer = {}
        for modality in self.datatypes:
            h, w = self.modalities_sizes[modality]
            if modality.startswith('img'): 
                self.buffer[modality] = ImageBuffer(h, w)
            elif modality.startswith('depth'):
                self.buffer[modality] = DepthBuffer(h, w)
            elif modality.startswith('disp'):
                self.buffer[modality] = DepthBuffer(h, w)
            elif modality.startswith('flow'):
                self.buffer[modality] = FlowBuffer(h, w)
            elif modality.startswith('fmask'):
                self.buffer[modality] = FMaskBuffer(h, w)
            else:
                assert False, "Unsupported Data Type {}".format(modality)

        self.trajlist, self.trajlenlist, self.framelist = [],[],[]
        self.full = False
        self.framenum = 0

    def reset(self, framenum, trajlist, trajlenlist, framelist):
        self.trajlist, self.trajlenlist, self.framelist = trajlist, trajlenlist, framelist
        for mod in self.datatypes:
            self.buffer[mod].reset(framenum)
        self.full = False
        self.framenum = framenum

    def insert(self, index, sample):
        '''
        sample: a dictionary
        {
            'mod0': n x h x w x c,
            'mod1': n x h x w x c, 
            ...
        }
        '''
        for mod in self.datatypes:
            datanp = sample[mod].numpy() # n x h x w x c
            for k in range(datanp.shape[0]):
                self.buffer[mod].insert(index+k, datanp[k])

    def __len__(self):
        return self.framenum

    def __getitem__(self, index):
        '''
        Note this function won't copy the data
        so do not modify this data! 
        '''
        sample = {}
        for mod in self.datatypes:
            sample[mod] = self.buffer[mod][index]
        return sample