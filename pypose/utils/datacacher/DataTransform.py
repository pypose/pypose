from xml.sax.xmlreader import InputSource
from .utils import DownscaleFlow, RandomCrop, RandomResizeCrop, RandomHSV, ToTensor, Normalize, Compose, FlipFlow, ResizeData, CropCenter, FlipStereo, RandomRotate, RandomUncertainty, FlowStereoNormalization

'''
Different datasets may need different transforms, especially the resizing-factor and cropping size
'''
class DataTransform(object):
    '''
    This class organizes the data augmentation for different training tasks
    This no longer handers the different requirements for different datasets
    '''
    def __init__(self) -> None:
        # self.args = args
        self.transformlist = []

    def random_resize_crop(self, resize_factor, image_height, image_width):
        # if self.args.no_data_augment:
        #     self.transformlist.append(CropCenter(size=(image_height, image_width)))
        #     return

        if resize_factor>0 : # and (self.datatype != 'kitti') kitti has sparse label which can not be resized. 
            self.transformlist.append(RandomResizeCrop(size=(image_height, image_width), max_scale=resize_factor))
        else:
            self.transformlist.append(RandomCrop(size=(image_height, image_width)))

    def resize_crop(self, resize_factor, image_height, image_width, crop_center=False):
        self.transformlist.append(ResizeData(size=0, fx=resize_factor, fy=resize_factor) )
        if crop_center:
            self.transformlist.append(CropCenter(size=(image_height, image_width)))
        else:
            self.transformlist.append(RandomCrop(size=(image_height,image_width)))

    def augment_color(self, hsv_rand):
        self.transformlist.append(RandomHSV((10,80,80), random_random=hsv_rand))

    def random_flip(self, mod):
        if mod == 'flow':
            self.transformlist.append(FlipFlow())
        elif mod == 'stereo':
            self.transformlist.append(FlipStereo())

    def downscale_flow(self):
        # if self.args.downscale_flow:
        self.transformlist.append(DownscaleFlow())

    def uncertainty(self, patchnum=5):
        # if self.args.uncertainty:
        # if self.args.test:
        self.transformlist.append(RandomUncertainty(patchnum=patchnum))
        # else: 
            # self.transformlist.append(RandomUncertainty())

    def normalize_to_tensor(self, normalize=True):
        if normalize:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            self.transformlist.append(Normalize(mean=mean,std=std))
        self.transformlist.append(ToTensor())

    def add_transformlist(self, transform):
        self.transformlist.append(transform)

    def get_transformlist(self, ):
        return Compose(self.transformlist)

class StereoDataTransform(DataTransform):
    def __init__(self, input_size, data_augment=False, resize_factor=1.0, rand_hsv=0.0, random_rotate_rightimg=0, stereo_norm_factor=1.0): 
        super().__init__()
        image_height, image_width = input_size
        if data_augment:
            self.random_resize_crop(resize_factor, image_height, image_width)
            self.augment_color(rand_hsv)
            self.random_flip(mod='stereo')
            if random_rotate_rightimg>0:
                self.add_transformlist(RandomRotate(maxangle=random_rotate_rightimg))
        else:
            self.resize_crop(resize_factor, image_height, image_width)
        if stereo_norm_factor != 1:
            self.add_transformlist(FlowStereoNormalization(norm_factor=stereo_norm_factor, mod='disp0'))
        self.normalize_to_tensor()


class FlowDataTransform(DataTransform):
    def __init__(self, input_size, data_augment=False, resize_factor=1.0, rand_hsv=0.0, flow_norm_factor=1.0):
        super().__init__()
        image_height, image_width = input_size
        # if data_augment:
        #     self.random_resize_crop(resize_factor, image_height, image_width)
        #     self.augment_color(rand_hsv)
        #     self.random_flip(mod='flow')
        # else:
        #     self.resize_crop(resize_factor, image_height, image_width)
        # if flow_norm_factor != 1:
        #     self.add_transformlist(FlowStereoNormalization(norm_factor=flow_norm_factor, mod='flow'))
        # self.normalize_to_tensor()

class FlowVODataTransform(DataTransform):
    def __init__(self, input_size, downscale_flow=False, uncertainty=False, resize_factor=1.0): 
        super().__init__()
        image_height, image_width = input_size
        self.random_resize_crop(resize_factor, image_height, image_width)
        if downscale_flow:
            self.downscale_flow()
        if uncertainty:
            self.uncertainty()
        self.normalize_to_tensor(normalize=False)

# class E2EVODataTransform(DataTransform):
#     def __init__(self, args):
#         super().__init__()
#         self.random_resize_crop()
#         # e2e-vo
#         # if args.random_intrinsic>0:
#         #     transformlist = [ RandomResizeCrop(size=(image_height,image_width), max_scale=args.random_intrinsic/320.0, 
#         #                                         keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
#         # else:
#         #     transformlist = [CropCenter((image_height, image_width), fix_ratio=args.fix_ratio, scale_w=args.scale_w, scale_disp=False)]
#         #     # transformlist = [ RandomCrop(size=(image_height,image_width)) ]

#         if datatype=='kitti':
#             transformlist = [ ResizeData(size=(image_height,1226)), RandomCrop(size=(image_height,image_width)) ] # hard code

#         # if args.downscale_flow:
#         #     from Datasets.utils import DownscaleFlow
#         #     transformlist.append(DownscaleFlow())

#         # stereo-flow-vo
#         # if args.random_intrinsic > 0:
#         #     transformlist = [RandomResizeCrop(size=(args.image_height, args.image_width), max_scale=args.random_intrinsic/320.0, 
#         #                                         keep_center=args.random_crop_center, fix_ratio=args.fix_ratio, scale_disp=False) ]
#         # else: # No augmentation
#         #     transformlist = [CropCenter((args.image_height, args.image_width), fix_ratio=args.fix_ratio, scale_w=args.scale_w, scale_disp=False)]

#         # if args.downscale_flow:    
#         #     from Datasets.utils import DownscaleFlow
#         #     transformlist.append(DownscaleFlow()) # TODO: is this correct to replace the ResizeData? 
#         #     # transformlist.append(ResizeData(size=(int(args.image_height/4), int(args.image_width/4))))

#         if args.uncertainty:
#             from Datasets.utils import RandomUncertainty
#             if args.test:
#                 transformlist.append(RandomUncertainty(patchnum=0))
#             else: 
#                 transformlist.append(RandomUncertainty())

#         if args.random_scale_disp_motion:
#             from Datasets.utils import RandomScaleDispMotion
#             transformlist.append(RandomScaleDispMotion())

#         if args.random_static>0.0:
#             from Datasets.utils import StaticMotion
#             transformlist.append(StaticMotion(Rate=args.random_static))

#         # transformlist.append(ToTensor())

#         # e2e stereo-vo
#         # if args.random_intrinsic>0:
#         #     transformlist = [ RandomResizeCrop(size=(image_height,image_width), max_scale=args.random_intrinsic/320.0, 
#         #                                         keep_center=args.random_crop_center, fix_ratio=args.fix_ratio, scale_disp=False) ]
#         # else:
#         #     transformlist = [CropCenter((args.image_height, args.image_width), fix_ratio=args.fix_ratio, scale_w=args.scale_w, scale_disp=False)]

#         # if args.downscale_flow:
#         #     from .utils import DownscaleFlow # without resize rgbs
#         #     transformlist.append(DownscaleFlow())

#         # if not args.no_data_augment:
#         #     transformlist.append(RandomHSV((10,80,80), random_random=args.hsv_rand))
#         # transformlist.extend([Normalize(mean=mean,std=std,keep_old=True),ToTensor()])

#         # imu
#         if args.imu_noise > 0.0: 
#             transformlist = [ IMUNoise(args.imu_noise)] 
#         else:
#             transformlist = []

#         if not args.no_imu_norm: 
#             transformlist.append(IMUNormalization())

#         # # flow-vio
#         # if args.random_intrinsic > 0:
#         #     transformlist = [RandomResizeCrop(size=(args.image_height, args.image_width), max_scale=args.random_intrinsic/320.0, 
#         #                                         keep_center=args.random_crop_center, fix_ratio=args.fix_ratio) ]
#         # else: # No augmentation
#         #     transformlist = [CropCenter((args.image_height, args.image_width))]

#         # if args.downscale_flow:    
#         #     from Datasets.utils import DownscaleFlow
#         #     transformlist.append(DownscaleFlow())

#         # if args.uncertainty:
#         #     if args.test:
#         #         transformlist.append(RandomUncertainty(patchnum=0))
#         #     else: 
#         #         transformlist.append(RandomUncertainty())
#         # transformlist.append(ToTensor())

#         if args.imu_noise > 0.0: 
#             transformlist.append(IMUNoise(args.imu_noise))
#         if not args.no_imu_norm: 
#             transformlist.append(IMUNormalization())        

        