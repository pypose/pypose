from setuptools import setup

setup(
    name='pypose', 
    version='0.0.1',
    description='PyPose = (GTSAM | G2O) x PyTorch',
    author='Chen Wang',
    packages=[
        'pypose',
        'pypose/liegroup',
        'pypose/module',
        'pypose/optim',
        'pypose/utils',
    ]
)