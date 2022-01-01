import os.path as osp
from setuptools import setup
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='pypose', 
    version='0.1.1',
    description='PyPose = (GTSAM | G2O) x PyTorch',
    author='Chen Wang',
    packages=['pypose'],
    license = 'BSD 3-Clause License',
    url = 'pypose.org',
    author_email = 'chenwang@dr.com',
    ext_modules=[
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'pypose/liegroup/lietorch/include'), 
                osp.join(ROOT, 'pypose/liegroup/eigen')],
            sources=[
                'pypose/liegroup/lietorch/src/lietorch.cpp', 
                'pypose/liegroup/lietorch/src/lietorch_gpu.cu',
                'pypose/liegroup/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'], 
                'nvcc': ['-O2',
                    '-gencode=arch=compute_60,code=sm_60', 
                    '-gencode=arch=compute_61,code=sm_61', 
                    '-gencode=arch=compute_70,code=sm_70', 
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_75,code=compute_75',
                ]
            }),

        CUDAExtension('lietorch_extras', 
            sources=[
                'pypose/liegroup/lietorch/extras/altcorr_kernel.cu',
                'pypose/liegroup/lietorch/extras/corr_index_kernel.cu',
                'pypose/liegroup/lietorch/extras/se3_builder.cu',
                'pypose/liegroup/lietorch/extras/se3_inplace_builder.cu',
                'pypose/liegroup/lietorch/extras/se3_solver.cu',
                'pypose/liegroup/lietorch/extras/extras.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O2'], 
                'nvcc': ['-O2',
                    '-gencode=arch=compute_60,code=sm_60', 
                    '-gencode=arch=compute_61,code=sm_61', 
                    '-gencode=arch=compute_70,code=sm_70', 
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_75,code=compute_75',
                    
                ]
            }),
    ],
    cmdclass={ 'build_ext': BuildExtension }
)
