## PyPose: A Library for Robot Learning with Physics-based Optimization

![robot](https://user-images.githubusercontent.com/8695500/193484553-2da66824-4461-4aca-ad8c-b17c05bef067.png)

-----

Deep learning has had remarkable success in robotic perception, but its data-centric nature suffers when it comes to generalizing to ever-changing environments. By contrast, physics-based optimization generalizes better, but it does not perform as well in complicated tasks due to the lack of high-level semantic information and the reliance on manual parametric tuning. To take advantage of these two complementary worlds, we present PyPose: a **robotics-oriented**, **PyTorch-based** library that combines **deep perceptual models** with **physics-based optimization techniques**. Our design goal for PyPose is to make it **user-friendly**, **efficient**, and **interpretable** with a tidy and well-organized architecture. Using an **imperative style interface**, it can be easily integrated into **real-world robotic applications**. 


-----

### Current Features

##### [LieTensor](https://pypose.org/docs/main/modules/)

- Lie group: [`SO3`](https://pypose.org/docs/main/generated/pypose.SO3/), [`SE3`](https://pypose.org/docs/main/generated/pypose.SE3/), [`Sim3`](https://pypose.org/docs/main/generated/pypose.Sim3/), [`RxSO3`](https://pypose.org/docs/main/generated/pypose.RxSO3/)
- Lie algebra: [`so3`](https://pypose.org/docs/main/generated/pypose.so3/), [`se3`](https://pypose.org/docs/main/generated/pypose.se3/), [`sim3`](https://pypose.org/docs/main/generated/pypose.sim3/), [`rxso3`](https://pypose.org/docs/main/generated/pypose.rxso3/)

##### [Modules](https://pypose.org/docs/main/modules/)

- System: [`LTI`](https://pypose.org/docs/main/generated/pypose.module.LTI), [`LTV`](https://pypose.org/docs/main/generated/pypose.module.LTV), [`NLS`](https://pypose.org/docs/main/generated/pypose.module.NLS)
- Filter: [`EKF`](https://pypose.org/docs/main/generated/pypose.module.EKF/), [`UKF`](https://pypose.org/docs/main/generated/pypose.module.UKF/), [`PF`](https://pypose.org/docs/main/generated/pypose.module.PF/)
- PnP Solver: [`EPnP`](https://pypose.org/docs/main/generated/pypose.module.EPnP/)
- Linear Quadratic Regulator: [`LQR`](https://pypose.org/docs/main/generated/pypose.module.LQR/)
- IMU Preintegration: [`IMUPreintegrator`](https://pypose.org/docs/main/generated/pypose.module.IMUPreintegrator/)
- ......

##### [Second-order Optimizers](https://pypose.org/docs/main/optim/)

- [`GaussNewton`](https://pypose.org/docs/main/generated/pypose.optim.GaussNewton)
- [`LevenbergMarquardt`](https://pypose.org/docs/main/generated/pypose.optim.LevenbergMarquardt/)
- ......

Want more features? [Create an issue here](https://github.com/pypose/pypose/issues) to requst new features.

##### PyPose is highly efficient and supports parallel computing for Jacobian of Lie group and Lie algebra. See following comparison.

<img width="1167" alt="image" src="https://user-images.githubusercontent.com/8695500/203210668-1a90224a-ae08-4d31-b9d1-e293be75ef3e.png">

Efficiency and memory comparison of batched Lie group operations (we take Theseus performance as 1×).

More information about efficiency comparison goes to [our paper for PyPose](https://arxiv.org/abs/2209.15428).

## Getting Started
    
### Installation

#### Install from **pypi**
```bash
pip install pypose
```

#### Install from source

1. Requirement:

On Ubuntu, MasOS, or Windows, install [PyTorch](https://pytorch.org/), then run:

```bash
pip install -r requirements/runtime.txt
```

2. Install locally:

```bash
git clone  https://github.com/pypose/pypose.git
cd pypose && python setup.py develop
```

3. Run tests

```bash
pytest
```

####  For contributors

1. Make sure the above installation is correct. 

2. Go to [CONTRIBUTING.md](CONTRIBUTING.md)


#### Examples

1. The following code sample shows how to rotate random points and compute the gradient of batched rotation.

```python
>>> import torch, pypose as pp

>>> # A random so(3) LieTensor
>>> r = pp.randn_so3(2, requires_grad=True)
    so3Type LieTensor:
    tensor([[ 0.1606,  0.0232, -1.5516],
            [-0.0807, -0.7184, -0.1102]], requires_grad=True)

>>> R = r.Exp() # Equivalent to: R = pp.Exp(r)
    SO3Type LieTensor:
    tensor([[ 0.0724,  0.0104, -0.6995,  0.7109],
            [-0.0395, -0.3513, -0.0539,  0.9339]], grad_fn=<AliasBackward0>)

>>> p = R @ torch.randn(3) # Rotate random point
    tensor([[ 0.8045, -0.8555,  0.5260],
            [ 0.3502,  0.8337,  0.9154]], grad_fn=<ViewBackward0>)

>>> p.sum().backward()     # Compute gradient
>>> r.grad                 # Print gradient
    tensor([[-0.7920, -0.9510,  1.7110],
            [-0.2659,  0.5709, -0.3855]])
```

2. This example shows how to estimate batched inverse of transform by a second-order optimizer. Two usage options for a `scheduler` are provided, each of which can work independently.

```python
>>> from torch import nn
>>> import torch, pypose as pp
>>> from pypose.optim import LM
>>> from pypose.optim.strategy import Constant
>>> from pypose.optim.scheduler import StopOnPlateau

>>> class InvNet(nn.Module):

        def __init__(self, *dim):
            super().__init__()
            init = pp.randn_SE3(*dim)
            self.pose = pp.Parameter(init)

        def forward(self, input):
            error = (self.pose @ input).Log()
            return error.tensor()

>>> device = torch.device("cuda")
>>> input = pp.randn_SE3(2, 2, device=device)
>>> invnet = InvNet(2, 2).to(device)
>>> strategy = Constant(damping=1e-4)
>>> optimizer = LM(invnet, strategy=strategy)
>>> scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=True)

>>> # 1st option, full optimization
>>> scheduler.optimize(input=input)

>>> # 2nd option, step optimization
>>> while scheduler.continual():
        loss = optimizer.step(input)
        scheduler.step(loss)

>>> # Note: remove one of the above options for usage!
```

For more usage, see [Documentation](https://pypose.org/docs). For more applications, see [Examples](https://github.com/pypose/pypose/tree/main/examples).

## Citing PyPose

If you use PyPose, please cite the paper below. You may also [download it here](https://arxiv.org/abs/2209.15428).

```bibtex
@inproceedings{wang2023pypose,
  title     = {{PyPose}: A Library for Robot Learning with Physics-based Optimization},
  author    = {Wang, Chen and Gao, Dasong and Xu, Kuan and Geng, Junyi and Hu, Yaoyu and Qiu, Yuheng and Li, Bowen and Yang, Fan and Moon, Brady and Pandey, Abhinav and Aryan and Xu, Jiahe and Wu, Tianhao and He, Haonan and Huang, Daning and Ren, Zhongqiang and Zhao, Shibo and Fu, Taimeng and Reddy, Pranay and Lin, Xiao and Wang, Wenshan and Shi, Jingnan and Talak, Rajat and Cao, Kun and Du, Yi and Wang, Han and Yu, Huai and Wang, Shanzhao and Chen, Siyu and Kashyap, Ananth  and Bandaru, Rohan and Dantu, Karthik and Wu, Jiajun and Xie, Lihua and Carlone, Luca and Hutter, Marco and Scherer, Sebastian},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}
```
