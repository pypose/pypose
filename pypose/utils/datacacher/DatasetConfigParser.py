from email.policy import default
from cv2 import transform
import yaml

from collections import OrderedDict

class DatasetConfigParser(object):
    """
    Class that reads in the spec dataset.
    """
    def __init__(self):
        # these params are required, assert error if not provided
        self.global_paramlist = ['task', 'modalities', 'input_seqlen']

        self.cacher_paramlist = [   'datatype', # which dataloader is used for cacher loading
                                    'load_size', # size in cacher
                                    'subset_framenum', # frame number in cacher
                                    'worker_num'] # how many works for the cacher

        self.transform_paramlist = ['data_augment', # whether use data augmentation
                                    'resize_factor', # data augment: RCR
                                    'input_size', # size for the training
                                    'rand_hsv', # data augment
                                    'flow_norm_factor',
                                    'stereo_norm_factor'
                                    ]

        self.dataset_paramlist = ['imu_freq',
                                  'intrinsics',
                                  'intrinsics_scale',
                                  'blxfx',
                                  'frame_skip',
                                  'seq_stride',
                                  'random_blur']

    def parse_from_fp(self, fp):
        x = yaml.safe_load(open(fp, 'r'))
        return self.parse(x)

    def parse(self, spec):
        dataset_config = OrderedDict()

        for param in self.global_paramlist:
            if param in spec: 
                dataset_config[param] = spec[param]
            else:
                assert False, "DatasetConfigParser: Missing {} in the spec file".format(param)

        default_cacher_params = {}
        for param in self.cacher_paramlist:
            key = 'cacher_'+param
            default_cacher_params[param]  = spec[key] if key in spec else None

        default_transform_params = {}
        for param in self.transform_paramlist:
            key = 'transform_'+param
            default_transform_params[param] = spec[key] if key in spec else None

        default_dataset_params = {}
        for param in self.dataset_paramlist:
            key = 'dataset_'+param
            default_dataset_params[param] = spec[key] if key in spec else None

        data_config = {}
        for datafile, params in spec['data'].items():
            all_params = {}

            cacher_params = {}
            assert 'cacher' in params, 'DatasetConfigParser: Missing cacher in the spec of {}'.format(datafile)
            cacherparams = params['cacher']
            for param in self.cacher_paramlist:
                if cacherparams is not None and param in cacherparams: # use specific param
                    cacher_params[param] = cacherparams[param]
                elif default_cacher_params[param] is not None: # use default param
                    cacher_params[param] = default_cacher_params[param]
            all_params['cacher'] = cacher_params

            # parse the parameters for the transform
            assert 'transform' in params, 'DatasetConfigParser: Missing transfrom in the spec of {}'.format(datafile)
            transparams = params['transform']
            trans_params = {}
            for param in self.transform_paramlist:
                if transparams is not None and param in transparams:
                    trans_params[param] = transparams[param]
                elif default_transform_params[param] is not None: # use default param
                    trans_params[param] = default_transform_params[param]
            all_params['transform'] = trans_params

            # parse the parameters for the transform
            assert 'dataset' in params, 'DatasetConfigParser: Missing dataset in the spec of {}'.format(datafile)
            datasetparams = params['dataset']
            dataset_params = {}
            for param in self.dataset_paramlist:
                if datasetparams is not None and param in datasetparams:
                    dataset_params[param] = datasetparams[param]
                elif default_dataset_params[param] is not None: # use default param
                    dataset_params[param] = default_dataset_params[param]
            all_params['dataset'] = dataset_params

            data_config[datafile] = all_params
        dataset_config['data'] = data_config
        return dataset_config


if __name__ == "__main__":
    fp = open('dataspec/flow_train_all.yaml')
    d = yaml.safe_load(fp)
    print(d)
    print(type(d))
    parser = DatasetConfigParser()
    x = parser.parse(d)
    print(x)
