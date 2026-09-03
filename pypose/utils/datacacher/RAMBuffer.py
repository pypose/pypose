import numpy as np
import ctypes
import multiprocessing as mp

def convert_type(nptype):
    '''
    return type, number of bytes
    '''
    if nptype == np.float32:
        return ctypes.c_float, 4
    if nptype == np.float64:
        return ctypes.c_double, 8
    if nptype == np.uint8:
        return ctypes.c_uint8, 1
    return None
 
class RAMBufferBase(object):
    # the buffer to store the sequential data
    def __init__(self, datatype):
        '''
        datatype: np datatype
        datasize: a tuple
        in general, the buffer is in the format of (n x h x w x c) or (n x h x w)
        '''
        self.ctype, self.databyte = convert_type(datatype)
        assert self.ctype is not None, "Type Error {}".format(datatype)

        self.datatype = datatype
        self.datasize = (0)
        # self.reset(datasize)

    def reset(self, datasize):
        if datasize != self.datasize: # re-allocate the buffer only if the datasize changes
            # print(datasize, self.datasize, datasize == self.datasize)
            datanum = int(np.prod(datasize))
            self.datasize = datasize
            buffer_base = mp.Array(self.ctype, datanum)
            self.buffer = np.ctypeslib.as_array(buffer_base.get_obj())
            self.buffer = self.buffer.reshape(self.datasize)
            print("RAM Buffer allocated size {}, mem {} G".format(datasize, datanum * self.databyte / 1000./1000./1000.))

    def insert(self, index, data):
        assert data.shape == self.datasize[1:], "Insert data shape error! Data shape {}, buffer shape {}".format(data.shape, self.datasize)
        assert data.dtype == self.datatype, "Insert data type error! Data type {}, buffer type {}".format(data.dtype, self.datatype)
        self.buffer[index] = data

    def __getitem__(self, index):
        # assert index < self.datasize[0], 'Invalid index {}, buffer size {}'.format(index, self.datasize[0])
        return self.buffer[index]

class ImageBuffer(RAMBufferBase):
    '''
    (framenum, 3, imgh, imgw)
    '''
    def __init__(self, imgh, imgw):
        super().__init__(np.uint8)
        self.imgh, self.imgw = imgh, imgw
    
    def reset(self, framenum):
        return super().reset((framenum, 3, self.imgh, self.imgw))

    def insert(self, index, data):
        data = data.transpose(2,0,1)
        return super().insert(index, data)

class DepthBuffer(RAMBufferBase):
    '''
    (framenum, imgh, imgw)
    '''
    def __init__(self, imgh, imgw):
        super().__init__(np.float32)
        self.imgh, self.imgw = imgh, imgw
    
    def reset(self, framenum):
        return super().reset((framenum, self.imgh, self.imgw))

class FlowBuffer(RAMBufferBase):
    '''
    (framenum, imgh, imgw, 2)
    '''
    def __init__(self, imgh, imgw):
        super().__init__(np.float32)
        self.imgh, self.imgw = imgh, imgw
    
    def reset(self, framenum):
        return super().reset((framenum, 2, self.imgh, self.imgw))

    def insert(self, index, data):
        data = data.transpose(2,0,1)
        return super().insert(index, data)

class FMaskBuffer(RAMBufferBase):
    '''
    (framenum, imgh, imgw)
    '''
    def __init__(self, imgh, imgw):
        super().__init__(np.uint8)
        self.imgh, self.imgw = imgh, imgw
    
    def reset(self, framenum):
        return super().reset((framenum, self.imgh, self.imgw))

if __name__=="__main__":
    rambuffer = RAMBufferBase(np.float32, (10,3,4,2))
    rambuffer.insert(0, np.random.rand(3,4,2))
    rambuffer.insert(3, np.random.rand(3,4,2))
    print(rambuffer.buffer)