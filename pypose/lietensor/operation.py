import torch
from .basics import vec2skew
from ..basics import pm, cumops, cummul, cumprod


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

def so3_adj(x):
    return vec2skew(x)

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

def se3_adj(x):
    adj_6x6 = torch.zeros((x.shape[:-1]+(6, 6)), device=x.device, dtype=x.dtype, requires_grad=False)
    Phi = vec2skew(x[..., 3:])
    adj_6x6[..., :3, :3] = Phi
    adj_6x6[..., :3, 3:] = vec2skew(x[..., :3])
    adj_6x6[..., 3:, 3:] = Phi
    return adj_6x6

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

def rxso3_adj(x):
    adj_4x4 = torch.zeros((x.shape[:-1]+(4, 4)), device=x.device, dtype=x.dtype, requires_grad=False)
    adj_4x4[..., :3, :3] = vec2skew(x[..., :3])
    return adj_4x4

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


def SO3_Adj(X):
    I3x3 = torch.eye(3, device=X.device, dtype=X.dtype).expand(X.shape[:-1]+(3, 3))
    Xv, Xw = X[..., :3], X[..., 3:]
    Xw_3x3 = Xw.unsqueeze(-1) * I3x3
    return 2.0 * Xw.unsqueeze(-1) * (Xw_3x3 + vec2skew(Xv)) - I3x3 + 2.0 * Xv.unsqueeze(-1) * Xv.unsqueeze(-2)


def SO3_Matrix(X):
    return SO3_Adj(X)


def SO3_Act_Jacobian(p):
    return vec2skew(-p)


def SO3_Matrix4x4(X):
    T = torch.eye(4, device=X.device, dtype=X.dtype, requires_grad=False).repeat(X.shape[:-1]+(1, 1))
    T[..., :3, :3] = SO3_Matrix(X)
    return T


def SO3_Act4_Jacobian(p):
    J = torch.zeros((p.shape[:-1]+(4, 3)), device=p.device, dtype=p.dtype, requires_grad=False)
    J[..., :3, :3] = SO3_Act_Jacobian(p[..., :3])
    return J


def SE3_Adj(X):
    Adj = torch.zeros((X.shape[:-1]+(6, 6)), device=X.device, dtype=X.dtype, requires_grad=False)
    t, q = X[..., :3], X[..., 3:]
    R3x3 = SO3_Adj(q)
    tx = vec2skew(t)
    Adj[..., :3, :3] = R3x3
    Adj[..., :3, 3:] = torch.matmul(tx, R3x3)
    Adj[..., 3:, 3:] = R3x3
    return Adj


def SE3_Matrix(X):
    T = torch.eye(4, device=X.device, dtype=X.dtype, requires_grad=False).repeat(X.shape[:-1]+(1, 1))
    T[..., :3, :3] = SO3_Matrix(X[..., 3:])
    T[..., :3, 3] = X[..., :3]
    return T


def SE3_Act_Jacobian(p):
    I3x3 = torch.eye(3, device=p.device, dtype=p.dtype).expand(p.shape[:-1]+(3, 3))
    return torch.cat((I3x3, vec2skew(-p)), dim=-1)


def SE3_Matrix4x4(X):
    return SE3_Matrix(X)


def SE3_Act4_Jacobian(p):
    J = torch.zeros((p.shape[:-1]+(4, 6)), device=p.device, dtype=p.dtype, requires_grad=False)
    I3x3 = torch.eye(3, device=p.device, dtype=p.dtype).expand(p.shape[:-1]+(3, 3))
    J[..., :3, :3] = I3x3 * p[..., 3:].unsqueeze(-1)
    J[..., :3, 3:] = vec2skew(-p[..., :3])
    return J


def RxSO3_Adj(X):
    Adj = torch.eye(4, device=X.device, dtype=X.dtype, requires_grad=False).repeat(X.shape[:-1]+(1, 1))
    Adj[..., :3, :3] = SO3_Adj(X[..., :4])
    return Adj


def RxSO3_Matrix(X):
    return X[..., 4:].unsqueeze(-1) * SO3_Adj(X[..., :4])


def RxSO3_Rotation(X):
    return SO3_Adj(X[..., :4])


def RxSO3_Act_Jacobian(p):
    return torch.cat((vec2skew(-p), p.unsqueeze(-1)), dim=-1)


def RxSO3_Matrix4x4(X):
    T = torch.eye(4, device=X.device, dtype=X.dtype, requires_grad=False).repeat(X.shape[:-1]+(1, 1))
    T[..., :3, :3] = RxSO3_Matrix(X)
    return T


def RxSO3_Act4_Jacobian(p):
    J = torch.zeros((p.shape[:-1]+(4, 4)), device=p.device, dtype=p.dtype, requires_grad=False)
    J[..., :3, :3] = SO3_Act_Jacobian(p[..., :3])
    J[..., :3, 3] = p[..., :3]
    return J


def Sim3_Adj(X):
    Adj = torch.eye(7, device=X.device, dtype=X.dtype, requires_grad=False).repeat(X.shape[:-1]+(1, 1))
    R = RxSO3_Rotation(X[..., 3:])
    tx = vec2skew(X[..., :3])
    Adj[..., :3, :3] = RxSO3_Matrix(X[..., 3:])
    Adj[..., :3, 3:6] = torch.matmul(tx, R)
    Adj[..., :3, 6] = -X[..., :3]
    Adj[..., 3:6, 3:6] = R
    return Adj


def Sim3_Matrix(X):
    T = torch.eye(4, device=X.device, dtype=X.dtype, requires_grad=False).repeat(X.shape[:-1]+(1, 1))
    T[..., :3, :3] = RxSO3_Matrix(X[..., 3:])
    T[..., :3, 3] = X[..., :3]
    return T


def Sim3_Act_Jacobian(p):
    return torch.cat((SE3_Act_Jacobian(p), p.unsqueeze(-1)), dim=-1)


def Sim3_Matrix4x4(X):
    T = torch.eye(4, device=X.device, dtype=X.dtype, requires_grad=False).repeat(X.shape[:-1]+(1, 1))
    T[..., :3, :3] = RxSO3_Matrix(X[..., 3:])
    T[..., :3, 3] = X[..., :3]
    return T


def Sim3_Act4_Jacobian(p):
    J = torch.zeros((p.shape[:-1]+(4, 7)), device=p.device, dtype=p.dtype, requires_grad=False)
    J[..., :6] = SE3_Act4_Jacobian(p)
    J[..., :3, 6] = p[..., :3]
    return J


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
        factor[idx2] = pm(w[idx2]) * torch.pi / v_norm[idx2]
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


class SO3_Act(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        Xv, Xw = X[..., :3], X[..., 3:]
        uv = torch.linalg.cross(Xv, p, dim=-1)
        uv += uv
        out = p + Xw * uv + torch.linalg.cross(Xv, uv, dim=-1)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        m = SO3_Matrix(X)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ SO3_Act_Jacobian(out)
        p_grad = dq @ m[..., :3, :3]
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class SE3_Act(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        out = X[..., :3] + SO3_Act.apply(X[..., 3:], p)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        m = SE3_Matrix(X)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ SE3_Act_Jacobian(out)
        p_grad = dq @ m[..., :3, :3]
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class RxSO3_Act(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        out = X[..., 4:] * SO3_Act.apply(X[..., :4], p)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        m = RxSO3_Matrix(X)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ RxSO3_Act_Jacobian(out)
        p_grad = dq @ m[..., :3, :3]
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class Sim3_Act(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        out = X[..., :3] + RxSO3_Act.apply(X[..., 3:], p)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        m = Sim3_Matrix(X)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ Sim3_Act_Jacobian(out)
        p_grad = dq @ m[..., :3, :3]
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class SO3_Act4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        out = torch.cat((SO3_Act.apply(X, p[..., :3]), p[..., 3:]), dim=-1)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ SO3_Act4_Jacobian(out)
        p_grad = dq @ SO3_Matrix4x4(X)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class SE3_Act4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        t = SO3_Act.apply(X[..., 3:], p[..., :3]) + X[..., :3] * p[..., 3:]
        out = torch.cat((t, p[..., 3:]), dim=-1)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ SE3_Act4_Jacobian(out)
        p_grad = dq @ SE3_Matrix4x4(X)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class RxSO3_Act4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        out = torch.cat((RxSO3_Act.apply(X, p[..., :3]), p[..., 3:]), dim=-1)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ RxSO3_Act4_Jacobian(out)
        p_grad = dq @ RxSO3_Matrix4x4(X)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class Sim3_Act4(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, p):
        t = RxSO3_Act.apply(X[..., 3:], p[..., :3]) + X[..., :3] * p[..., 3:]
        out = torch.cat((t, p[..., 3:]), dim=-1)
        ctx.save_for_backward(X, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, out = ctx.saved_tensors
        dq = grad_output.unsqueeze(-2)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = dq @ Sim3_Act4_Jacobian(out)
        p_grad = dq @ Sim3_Matrix4x4(X)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), p_grad.squeeze(-2)


class SO3_AdjXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        adj_matrix = SO3_Adj(X)
        out = (adj_matrix @ a.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(out, adj_matrix)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, adj_matrix = ctx.saved_tensors
        X_grad = -grad_output.unsqueeze(-2) @ so3_adj(out)
        a_grad = grad_output.unsqueeze(-2) @ adj_matrix
        zero = torch.zeros(out.shape[:-1]+(1,), device=out.device, dtype=out.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad.squeeze(-2)


class SE3_AdjXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        adj_matrix = SE3_Adj(X)
        out = (adj_matrix @ a.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(out, adj_matrix)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, adj_matrix = ctx.saved_tensors
        X_grad = -grad_output.unsqueeze(-2) @ se3_adj(out)
        a_grad = grad_output.unsqueeze(-2) @ adj_matrix
        zero = torch.zeros(out.shape[:-1]+(1,), device=out.device, dtype=out.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad.squeeze(-2)


class RxSO3_AdjXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        adj_matrix = RxSO3_Adj(X)
        out = (adj_matrix @ a.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(out, adj_matrix)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, adj_matrix = ctx.saved_tensors
        X_grad = -grad_output.unsqueeze(-2) @ rxso3_adj(out)
        a_grad = grad_output.unsqueeze(-2) @ adj_matrix
        zero = torch.zeros(out.shape[:-1]+(1,), device=out.device, dtype=out.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad.squeeze(-2)


class Sim3_AdjXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        adj_matrix = Sim3_Adj(X)
        out = (adj_matrix @ a.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(out, adj_matrix)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, adj_matrix = ctx.saved_tensors
        X_grad = -grad_output.unsqueeze(-2) @ sim3_adj(out)
        a_grad = grad_output.unsqueeze(-2) @ adj_matrix
        zero = torch.zeros(out.shape[:-1]+(1,), device=out.device, dtype=out.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad.squeeze(-2)


class SO3_Mul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X)
        Xv, Xw, Yv, Yw = X[..., :3], X[..., 3:], Y[..., :3], Y[..., 3:]
        Zv = Xw * Yv + Xv * Yw + torch.linalg.cross(Xv, Yv, dim=-1)
        Zw = Xw * Yw - (Xv * Yv).sum(dim=-1, keepdim=True)
        return torch.cat([Zv, Zw], -1)

    @staticmethod
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = torch.cat((grad_output[..., :-1], zero), dim = -1)
        dZxX = torch.matmul(grad_output[..., :-1].unsqueeze(-2), SO3_Adj(X)).squeeze(-2) 
        Y_grad = torch.cat((dZxX, zero), dim = -1)
        return X_grad, Y_grad


class SE3_Mul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X)
        t = X[..., :3] + SO3_Act.apply(X[..., 3:], Y[..., :3])
        q = SO3_Mul.apply(X[..., 3:], Y[..., 3:])
        return torch.cat((t, q), -1)

    @staticmethod
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = torch.cat((grad_output[..., :-1], zero), dim = -1)
        dZxX = torch.matmul(grad_output[..., :-1].unsqueeze(-2), SE3_Adj(X)).squeeze(-2) 
        Y_grad = torch.cat((dZxX, zero), dim = -1)
        return X_grad, Y_grad        


class RxSO3_Mul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X)
        q = SO3_Mul.apply(X[..., :4], Y[..., :4])
        s = X[..., 4:] * Y[..., 4:]
        return torch.cat((q, s), -1)

    @staticmethod
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = torch.cat((grad_output[..., :-1], zero), dim = -1)
        dZxX = torch.matmul(grad_output[..., :-1].unsqueeze(-2), RxSO3_Adj(X)).squeeze(-2) 
        Y_grad = torch.cat((dZxX, zero), dim = -1)
        return X_grad, Y_grad    


class Sim3_Mul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, Y):
        ctx.save_for_backward(X)
        t = X[..., :3] + RxSO3_Act.apply(X[..., 3:], Y[..., :3])
        q = RxSO3_Mul.apply(X[..., 3:], Y[..., 3:])
        return torch.cat((t, q), -1)

    @staticmethod
    def backward(ctx, grad_output):
        X = ctx.saved_tensors[0]
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        X_grad = torch.cat((grad_output[..., :-1], zero), dim = -1)
        dZxX = torch.matmul(grad_output[..., :-1].unsqueeze(-2), Sim3_Adj(X)).squeeze(-2) 
        Y_grad = torch.cat((dZxX, zero), dim = -1)
        return X_grad, Y_grad   


class SO3_Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        Y = torch.cat((-X[..., :3], X[..., 3:]), -1)
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        Y = ctx.saved_tensors[0]
        X_grad = -(grad_output[..., :-1].unsqueeze(-2) @ SO3_Adj(Y)).squeeze(-2)
        zero = torch.zeros(Y.shape[:-1]+(1,), device=Y.device, dtype=Y.dtype)
        return torch.cat((X_grad, zero), dim = -1)


class SE3_Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        q_inv = SO3_Inv.apply(X[..., 3:]) 
        t_inv = -SO3_Act.apply(q_inv, X[..., :3])
        Y = torch.cat((t_inv, q_inv), dim = -1)
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        Y = ctx.saved_tensors[0]
        X_grad = -(grad_output[..., :-1].unsqueeze(-2) @ SE3_Adj(Y)).squeeze(-2)
        zero = torch.zeros(Y.shape[:-1]+(1,), device=Y.device, dtype=Y.dtype)
        return torch.cat((X_grad, zero), dim = -1)


class RxSO3_Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        q_inv = SO3_Inv.apply(X[..., :4]) 
        s_inv = 1.0 / X[..., 4:]
        Y = torch.cat((q_inv, s_inv), dim = -1)
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        Y = ctx.saved_tensors[0]
        X_grad = -(grad_output[..., :-1].unsqueeze(-2) @ RxSO3_Adj(Y)).squeeze(-2)
        zero = torch.zeros(Y.shape[:-1]+(1,), device=Y.device, dtype=Y.dtype)
        return torch.cat((X_grad, zero), dim = -1)


class Sim3_Inv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        qs_inv = RxSO3_Inv.apply(X[..., 3:]) 
        t_inv = -RxSO3_Act.apply(qs_inv, X[..., :3])
        Y = torch.cat((t_inv, qs_inv), dim = -1)
        ctx.save_for_backward(Y)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        Y = ctx.saved_tensors[0]
        X_grad = -(grad_output[..., :-1].unsqueeze(-2) @ Sim3_Adj(Y)).squeeze(-2)
        zero = torch.zeros(Y.shape[:-1]+(1,), device=Y.device, dtype=Y.dtype)
        return torch.cat((X_grad, zero), dim = -1)


class SO3_AdjTXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        ctx.save_for_backward(X, a)
        out = SO3_AdjXa.apply(SO3_Inv.apply(X), a)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, a = ctx.saved_tensors
        a_grad = SO3_AdjXa.apply(X, grad_output)
        X_grad = -a.unsqueeze(-2) @ so3_adj(a_grad)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad


class SE3_AdjTXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        ctx.save_for_backward(X, a)
        out = SE3_AdjXa.apply(SE3_Inv.apply(X), a)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, a = ctx.saved_tensors
        a_grad = SE3_AdjXa.apply(X, grad_output)
        X_grad = -a.unsqueeze(-2) @ se3_adj(a_grad)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad


class RxSO3_AdjTXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        ctx.save_for_backward(X, a)
        out = RxSO3_AdjXa.apply(RxSO3_Inv.apply(X), a)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, a = ctx.saved_tensors
        a_grad = RxSO3_AdjXa.apply(X, grad_output)
        X_grad = -a.unsqueeze(-2) @ rxso3_adj(a_grad)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad


class Sim3_AdjTXa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, a):
        ctx.save_for_backward(X, a)
        out = Sim3_AdjXa.apply(Sim3_Inv.apply(X), a)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        X, a = ctx.saved_tensors
        a_grad = Sim3_AdjXa.apply(X, grad_output)
        X_grad = -a.unsqueeze(-2) @ sim3_adj(a_grad)
        zero = torch.zeros(X.shape[:-1]+(1,), device=X.device, dtype=X.dtype)
        return torch.cat((X_grad.squeeze(-2), zero), dim = -1), a_grad


def broadcast_inputs(x, y):
    """ Automatic broadcasting of missing dimensions """
    if y is None:
        xs, xd = x.shape[:-1], x.shape[-1]
        return (x.reshape(-1, xd).contiguous(), ), x.shape[:-1]
    out_shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    shape = out_shape if out_shape != torch.Size([]) else (1,)
    x = x.expand(shape+(x.shape[-1],)).reshape(-1,x.shape[-1]).contiguous()
    y = y.expand(shape+(y.shape[-1],)).reshape(-1,y.shape[-1]).contiguous()
    return (x, y), tuple(out_shape)