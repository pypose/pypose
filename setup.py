import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='pypose',
    version='0.1.1',
    description='To connect classic robotics with modern learning methods.',
    author='Chen Wang',
    packages=find_packages(),
    license = 'BSD 3-Clause License',
    url = 'pypose.org',
    author_email = 'chenwang@dr.com',
    ext_modules=[
        CUDAExtension('lietensor_backends', 
            include_dirs=[
                osp.join(ROOT, 'pypose/lietensor/include'),
                osp.join(ROOT, 'pypose/eigen')],
            sources=[
                'pypose/lietensor/src/lietensor.cpp',
                'pypose/lietensor/src/lietensor_gpu.cu',
                'pypose/lietensor/src/lietensor_cpu.cpp'],
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

        CUDAExtension('lietensor_extras',
            sources=[
                'pypose/lietensor/extras/altcorr_kernel.cu',
                'pypose/lietensor/extras/corr_index_kernel.cu',
                'pypose/lietensor/extras/se3_builder.cu',
                'pypose/lietensor/extras/se3_inplace_builder.cu',
                'pypose/lietensor/extras/se3_solver.cu',
                'pypose/lietensor/extras/extras.cpp',
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
