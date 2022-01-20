import torch
import liegroup_backends as backend


class GroupOp(torch.autograd.Function):
    """ group operation base class """

    @classmethod
    def forward(cls, ctx, group_id, *inputs):
        ctx.group_id = group_id
        ctx.save_for_backward(*inputs)
        out = cls.forward_op(ctx.group_id, *inputs)
        return out

    @classmethod
    def backward(cls, ctx, grad):
        error_str = "Backward operation not implemented for {}".format(cls)
        assert cls.backward_op is not None, error_str

        inputs = ctx.saved_tensors
        grad = grad.contiguous()
        grad_inputs = cls.backward_op(ctx.group_id, grad, *inputs)
        return (None, ) + tuple(grad_inputs)

class exp(GroupOp):
    """ exponential map """
    forward_op, backward_op = backend.expm, backend.expm_backward

class log(GroupOp):
    """ logarithm map """
    forward_op, backward_op = backend.logm, backend.logm_backward

class inv(GroupOp):
    """ group inverse """
    forward_op, backward_op = backend.inv, backend.inv_backward

class mul(GroupOp):
    """ group multiplication """
    forward_op, backward_op = backend.mul, backend.mul_backward

class adj(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = backend.adj, backend.adj_backward

class adjT(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = backend.adjT, backend.adjT_backward

class act3(GroupOp):
    """ action on point """
    forward_op, backward_op = backend.act, backend.act_backward

class act4(GroupOp):
    """ action on point """
    forward_op, backward_op = backend.act4, backend.act4_backward

class jinv(GroupOp):
    """ adjoint operator """
    forward_op, backward_op = backend.Jinv, None

class toMatrix(GroupOp):
    """ convert to matrix representation """
    forward_op, backward_op = backend.as_matrix, None
