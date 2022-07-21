import math, numbers
import xdrlib
import torch, warnings
from torch import nn, linalg
from torch.autograd import Function

from .basics import vec2skew, cumops, cummul, cumprod


def so3_Jl(x):
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
    return (I + coef1 * K + coef2 * (K@K))


def so3_Jl_inv(x):
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


def calcQ(x):
    tau, phi = x[..., :3], x[..., 3:]
    Tau, Phi = vec2skew(tau), vec2skew(phi)
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).unsqueeze(-1)
    theta2 = theta**2
    theta4 = theta2**2
    idx = (theta > torch.finfo(theta.dtype).eps)
    # coef1
    coef1 = torch.zeros_like(theta, requires_grad=False)
    coef1[idx] = (theta[idx] - theta[idx].sin()) / (theta2[idx] * theta[idx])
    coef1[~idx] = 1.0 / 6.0 - (1.0 / 120.0) * theta2[~idx] 
    # coef2
    coef2 = torch.zeros_like(theta, requires_grad=False)
    coef2[idx] = (theta2[idx] + 2 * theta[idx].cos() - 2) / (2 * theta4[idx])
    coef2[~idx] = 1.0 / 24.0 - (1.0 / 720.0) * theta2[~idx] 
    # coef3
    coef3 = torch.zeros_like(theta, requires_grad=False)
    coef3[idx] = (2 * theta[idx] - 3 * theta[idx].sin() + theta[idx] * theta[idx].cos()) / (2 * theta4[idx] * theta[idx])
    coef3[~idx] = 1.0 / 120.0 - (1.0 / 2520.0) * theta2[~idx] 
    Q = 0.5 * Tau + coef1 * (Phi@Tau + Tau@Phi + Phi@Tau@Phi) + \
        coef2 * (Phi@Phi@Tau + Tau@Phi@Phi - 3*Phi@Tau@Phi) + coef3 * (Phi@Tau@Phi@Phi + Phi@Phi@Tau@Phi)
    return Q


def se3_Jl(x):
    Zero3x3 = torch.zeros(3, 3, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(3, 3))
    J = so3_Jl(x[..., 3:])
    J6x6 = torch.cat((torch.cat((J, calcQ(x)), dim = -1), torch.cat((Zero3x3, J), dim = -1)), dim = -2)
    return J6x6


def se3_Jl_inv(x):
    Jl_inv_3x3, Q = so3_Jl_inv(x[..., 3:]), calcQ(x)
    Jl_inv_6x6 = torch.zeros((x.shape[:-1]+(6, 6)), device=x.device, dtype=x.dtype, requires_grad=False)
    Jl_inv_6x6[..., :3, :3] = Jl_inv_3x3
    Jl_inv_6x6[..., :3, 3:] = -Jl_inv_3x3 @ Q @ Jl_inv_3x3
    Jl_inv_6x6[..., 3:, 3:] = Jl_inv_3x3
    return Jl_inv_6x6


def rxso3_Ws(x):
    rotation, sigma = x[..., :3], x[..., 3]
    theta = torch.norm(rotation, 2, -1)

    A = torch.zeros_like(theta, requires_grad=False)
    B = torch.zeros_like(theta, requires_grad=False)
    C = torch.zeros_like(theta, requires_grad=False)

    sigma_larger = (sigma.abs() > torch.finfo(sigma.dtype).eps)
    theta_larger = (theta > torch.finfo(theta.dtype).eps)
    condition1 = (~sigma_larger) & (~theta_larger)
    condition2 = (~sigma_larger) & theta_larger
    condition3 = sigma_larger & (~theta_larger)
    condition4 = sigma_larger & theta_larger

    scale, sigma2, theta2 = sigma.exp(), sigma * sigma, theta * theta
    theta2_inv = 1.0 / theta2

    # condition1
    C[(~sigma_larger)], A[condition1], B[condition1] = 1.0, 0.5, 1.0 / 6

    # condition2
    theta_c2 = theta[condition2]      
    A[condition2] = (1.0 - theta_c2.cos()) * theta2_inv[condition2]
    B[condition2] = (theta_c2 - theta_c2.sin()) / (theta2[condition2] * theta_c2)

    # condition3        
    C[sigma_larger] = (scale[sigma_larger] - 1.0) / sigma[sigma_larger]
    sigma_c3, scale_c3, sigma2_c3 = sigma[condition3], scale[condition3], sigma2[condition3]
    A[condition3] = (1.0 + (sigma_c3 - 1.0) * scale_c3) / sigma2_c3
    B[condition3] = (0.5 * sigma2_c3 * scale_c3 + scale_c3 - 1.0 - sigma2_c3 * scale_c3) / (sigma2_c3 * sigma_c3)

    # condition4
    sigma_c4, sigma2_c4, scale_c4 = sigma[condition4], sigma2[condition4], scale[condition4]
    theta_c4, theta2_c4, theta2_inv_c4 = theta[condition4], theta2[condition4], theta2_inv[condition4]
    a_c4, b_c4, c_c4 = scale_c4 * theta_c4.sin(), scale_c4 * theta_c4.cos(), (theta2_c4 + sigma2_c4)
    A[condition4] = (a_c4 * sigma_c4 + (1 - b_c4) * theta_c4) / (theta_c4 * c_c4)
    B[condition4] = (C[condition4] - ((b_c4 - 1) * sigma_c4 + a_c4 * theta_c4) / c_c4) * theta2_inv_c4

    K = vec2skew(rotation)
    A = A.unsqueeze(-1).unsqueeze(-1)
    B = B.unsqueeze(-1).unsqueeze(-1)
    C = C.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(3,3))
    return A * K + B * (K@K) + C * I


def rxso3_Jl(x):
    J = torch.eye(4, device=x.device, dtype=x.dtype).repeat(x.shape[:-1]+(1, 1))
    J[..., :3, :3] = so3_Jl(x[..., :3])
    return J


def rxso3_Jl_inv(x):
    J_inv = torch.eye(4, device=x.device, dtype=x.dtype).repeat(x.shape[:-1]+(1, 1))
    J_inv[..., :3, :3] = so3_Jl_inv(x[..., :3])
    return J_inv


def sim3_adj(x):
    tau, phi, sigma = x[..., :3], x[..., 3:6], x[..., 6:]
    Tau, Phi = vec2skew(tau), vec2skew(phi)
    I3x3 = torch.eye(3, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(3, 3))
    ad = torch.zeros((x.shape[:-1]+(7, 7)), device=x.device, dtype=x.dtype, requires_grad=False)
    ad[..., :3, :3] = Phi + sigma.unsqueeze(-1) * I3x3
    ad[..., :3, 3:6] = Tau
    ad[..., :3, 6] = -tau
    ad[..., 3:6, 3:6] = Phi
    return ad


def sim3_Jl(x):
    Xi = sim3_adj(x)
    Xi2 = Xi @ Xi
    Xi4 = Xi2 @ Xi2
    I7x7 = torch.eye(7, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(7, 7))
    return (I7x7 + (1.0/2.0) * Xi + (1.0/6.0) * Xi2 + (1.0/24.0) * Xi @ Xi2 + (1.0/120.0) * Xi4 + (1.0/720.0) * Xi @ Xi4)


def sim3_Jl_inv(x):
    Xi = sim3_adj(x)
    Xi2 = Xi @ Xi
    Xi4 = Xi2 @ Xi2
    I7x7 = torch.eye(7, device=x.device, dtype=x.dtype).expand(x.shape[:-1]+(7, 7))
    return (I7x7 - (1.0/2.0) * Xi + (1.0/12.0) * Xi2 - (1.0/720.0) * Xi4)


class SO3_Log(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        eps = torch.finfo(input.dtype).eps
        v, w = input[..., :3], input[..., 3:]
        v_norm = torch.norm(v, 2, dim=-1, keepdim=True)
        w_abs = torch.abs(w)
        v_larger_than_eps = (v_norm > eps)
        w_larger_than_eps = (w_abs > eps)
        idx1 = v_larger_than_eps & w_larger_than_eps
        idx2 = v_larger_than_eps & (~w_larger_than_eps)
        idx3 = (~v_larger_than_eps)

        factor = torch.zeros_like(v_norm, requires_grad=False)
        factor[idx1] = 2.0 * torch.atan(v_norm[idx1]/w[idx1]) / v_norm[idx1]
        factor[idx2] = torch.sign(w[idx2]) * torch.pi / v_norm[idx2]
        factor[idx3] = 2.0 * (1.0 / w[idx3] - v_norm[idx3] * v_norm[idx3] / (3 * w[idx3]**3))
        output = factor * v
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        Jl_inv = so3_Jl_inv(output)
        grad = (grad_output.unsqueeze(-2) @ Jl_inv).squeeze(-2)
        zero = torch.zeros(output.shape[:-1]+(1,), device=output.device, dtype=output.dtype)
        return torch.cat((grad, zero), dim=-1)


class so3_Exp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
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
        input = ctx.saved_tensors[0]
        Jl = so3_Jl(input)
        grad_input = grad_output[..., :-1].unsqueeze(-2) @ Jl
        return grad_input.squeeze(-2)


class SE3_Log(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        phi = SO3_Log.apply(input[..., 3:])
        Jl_inv = so3_Jl_inv(phi)
        tau = (Jl_inv @ input[..., :3].unsqueeze(-1)).squeeze(-1)
        output = torch.cat([tau, phi], -1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        Jl_inv = se3_Jl_inv(output)
        grad = (grad_output.unsqueeze(-2) @ Jl_inv).squeeze(-2)
        zero = torch.zeros(output.shape[:-1]+(1,), device=output.device, dtype=output.dtype)
        return torch.cat((grad, zero), dim=-1)


class se3_Exp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        t = (so3_Jl(input[..., 3:]) @ input[..., :3].unsqueeze(-1)).squeeze(-1)
        r = so3_Exp.apply(input[..., 3:])
        return torch.cat([t, r], -1)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        Jl = se3_Jl(input)
        grad_input = grad_output[..., :-1].unsqueeze(-2) @ Jl
        return grad_input.squeeze(-2)


class RxSO3_Log(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        phi = SO3_Log.apply(input[..., :4])
        output = torch.cat([phi, input[..., 4:].log()], -1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        Jl_inv = rxso3_Jl_inv(output)
        grad = (grad_output.unsqueeze(-2) @ Jl_inv).squeeze(-2)
        zero = torch.zeros(output.shape[:-1]+(1,), device=output.device, dtype=output.dtype)
        return torch.cat((grad, zero), dim=-1)


class rxso3_Exp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        r = so3_Exp.apply(input[..., :3])
        s = torch.exp(input[..., 3:])
        return torch.cat([r, s], -1)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        Jl = rxso3_Jl(input)
        grad_input = grad_output[..., :-1].unsqueeze(-2) @ Jl
        return grad_input.squeeze(-2)


class Sim3_Log(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        phi_sigma = RxSO3_Log.apply(input[..., 3:])
        Ws_inv = rxso3_Ws(phi_sigma).inverse()
        tau = (Ws_inv @ input[..., :3].unsqueeze(-1)).squeeze(-1)
        output = torch.cat([tau, phi_sigma], -1)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        Jl_inv = sim3_Jl_inv(output)
        grad = (grad_output.unsqueeze(-2) @ Jl_inv).squeeze(-2)
        zero = torch.zeros(output.shape[:-1]+(1,), device=output.device, dtype=output.dtype)
        return torch.cat((grad, zero), dim=-1)


class sim3_Exp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        Ws = rxso3_Ws(input[..., 3:])
        t = (Ws @ input[..., :3].unsqueeze(-1)).squeeze(-1)
        r = rxso3_Exp.apply(input[..., 3:])
        return torch.cat([t, r], -1)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        Jl = sim3_Jl(input)
        grad_input = grad_output[..., :-1].unsqueeze(-2) @ Jl
        return grad_input.squeeze(-2)