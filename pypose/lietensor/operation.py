import math, numbers
import torch, warnings
from torch import nn, linalg
from torch.autograd import Function

from .basics import vec2skew, cumops, cummul, cumprod

def so3_Jl(self, x):
    K = vec2skew(x)
    theta = torch.linalg.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
    theta2 = theta**2
    I = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(3, 3))
    idx = (theta > torch.finfo(theta.dtype).eps)
    coef1 = torch.zeros_like(theta, requires_grad=False)
    coef1[idx] = (1-theta[idx].cos())/theta2[idx]
    coef1[~idx] = 0.5 - (1.0/24.0) * theta2[~idx]

    coef2 = torch.zeros_like(theta, requires_grad=False)
    coef2[idx] = (theta[idx] - theta[idx].sin()) / (theta[idx] * theta2[idx])
    coef2[~idx] = 1.0/6.0 - (1.0/120) * theta2[~idx]
    return (I - coef1 * K + coef2 * (K@K))


def Jl_inv(self, x):
    """
    Left jocobian inverse of SO(3)
    """
    K = vec2skew(x)
    theta = torch.linalg.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
    I = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(3, 3))
    idx = (theta > torch.finfo(theta.dtype).eps)
    coef2 = torch.zeros_like(theta, requires_grad=False)
    theta_idx = theta[idx]
    theta_half_idx, theta2_idx = 0.5 * theta_idx, theta_idx * theta_idx
    coef2[idx] = (1.0 - theta_idx * theta_half_idx.cos() / (2.0 * theta_half_idx.sin())) / theta2_idx
    coef2[~idx] = 1.0 / 12.0
    return (I - 0.5 * K + coef2 * (K @ K))


class so3_Exp(Function):

  @staticmethod
  def froward(ctx, input):
    ctx.save_for_backward(input)
    theta = torch.norm(input, 2, dim=-1, keepdim=True)
    theta_half, theta2 = 0.5 * theta, theta * theta
    theta4 = theta2 * theta2

    imag_factor = torch.zeros_like(theta, requires_grad=False)
    real_factor = torch.zeros_like(theta, requires_grad=False)
    idx = (theta > torch.finfo(theta.dtype).eps)
    imag_factor[idx] = torch.sin(theta_half[idx]) / theta[idx]
    real_factor[idx] = torch.cos(theta_half[idx])
    imag_factor[~idx] = 0.5 - (1.0/48.0) * theta2[~idx] + (1.0/3840.0) * theta4[~idx]
    real_factor[~idx] = 1.0 - (1.0/8.0) * theta2[~idx] + (1.0/384.0) * theta4[~idx]

    return torch.cat([input * imag_factor, real_factor], -1)

  @staticmethod
  def backward(ctx, grad_output):
    input = ctx.saved_tensors
