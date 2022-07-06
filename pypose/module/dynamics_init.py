import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

from . import util

ACTS = {
    'sigmoid': torch.sigmoid,
    'relu': F.relu,
    'elu': F.elu,
}

class NNDynamics(nn.Module):
    def __init__(self, n_state, n_ctrl, hidden_sizes=[100],
                 activation='sigmoid', passthrough=True):
        super().__init__()

        self.passthrough = passthrough

        self.fcs = []
        in_sz = n_state+n_ctrl
        for out_sz in hidden_sizes + [n_state]:
            fc = nn.Linear(in_sz, out_sz)
            self.fcs.append(fc)
            in_sz = out_sz
        self.fcs = nn.ModuleList(self.fcs)

        assert activation in ACTS.keys()
        act_f = ACTS[activation]
        self.activation = activation
        self.acts = [act_f]*(len(self.fcs)-1)+[lambda x:x] # Activation functions.

        self.Ws = [y.weight for y in self.fcs]
        self.zs = [] # Activations.


    def __getstate__(self):
        return (self.fcs, self.activation, self.passthrough)


    def __setstate__(self, state):
        super().__init__()
        if len(state) == 2:
            # TODO: Remove this soon, keeping for some old models.
            self.fcs, self.activation = state
            self.passthrough = True
        else:
            self.fcs, self.activation, self.passthrough = state

        act_f = ACTS[self.activation]
        self.acts = [act_f]*(len(self.fcs)-1)+[lambda x:x] # Activation functions.
        self.Ws = [y.weight for y in self.fcs]


    def forward(self, x, u):
        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)

        self.zs = []
        z = torch.cat((x, u), 1)
        for act, fc in zip(self.acts, self.fcs):
            z = act(fc(z))
            self.zs.append(z)

        # Hack: Don't include the output.
        self.zs = self.zs[:-1]

        if self.passthrough:
            z += x

        if x_dim == 1:
            z = z.squeeze(0)

        return z

    def grad_input(self, x, u):
        assert isinstance(x, Variable) == isinstance(u, Variable)
        diff = isinstance(x, Variable)

        x_dim, u_dim = x.ndimension(), u.ndimension()
        n_batch, n_state = x.size()
        _, n_ctrl = u.size()

        if not diff:
            Ws = [W.data for W in self.Ws]
            zs = [z.data for z in self.zs]
        else:
            Ws = self.Ws
            zs = self.zs

        assert len(zs) == len(Ws)-1
        grad = Ws[-1].repeat(n_batch,1,1)
        for i in range(len(zs)-1, 0-1, -1):
            n_out, n_in = Ws[i].size()

            if self.activation == 'relu':
                I = util.get_data_maybe(zs[i] <= 0.).unsqueeze(2).repeat(1,1,n_in)
                Wi_grad = Ws[i].repeat(n_batch,1,1)
                Wi_grad[I] = 0.
            elif self.activation == 'sigmoid':
                d = zs[i]*(1.-zs[i])
                d = d.unsqueeze(2).expand(n_batch, n_out, n_in)
                Wi_grad = Ws[i].repeat(n_batch,1,1)*d
            else:
                assert False

            grad = grad.bmm(Wi_grad)

        R = grad[:,:,:n_state]
        S = grad[:,:,n_state:]

        if self.passthrough:
            I = torch.eye(n_state).type_as(util.get_data_maybe(R)) \
                .unsqueeze(0).repeat(n_batch, 1, 1)

            if diff:
                I = Variable(I)

            R = R + I

        if x_dim == 1:
            R = R.squeeze(0)
            S = S.squeeze(0)

        return R, S


class CtrlPassthroughDynamics(nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self.dynamics = dynamics

    def forward(self, tilde_x, u):
        tilde_x_dim, u_dim = tilde_x.ndimension(), u.ndimension()
        if tilde_x_dim == 1:
            tilde_x = tilde_x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)

        n_ctrl = u.size(1)
        x = tilde_x[:,n_ctrl:]
        xtp1 = self.dynamics(x, u)
        tilde_xtp1 = torch.cat((u, xtp1), dim=1)

        if tilde_x_dim == 1:
            tilde_xtp1 = tilde_xtp1.squeeze()

        return tilde_xtp1

    def grad_input(self, x, u):
        assert False, "Unimplemented"


class AffineDynamics(nn.Module):
    def __init__(self, A, B, c=None):
        super(AffineDynamics, self).__init__()

        assert A.ndimension() == 2
        assert B.ndimension() == 2
        if c is not None:
            assert c.ndimension() == 1

        self.A = A
        self.B = B
        self.c = c

    def forward(self, x, u):
        if not isinstance(x, Variable) and isinstance(self.A, Variable):
            A = self.A.data
            B = self.B.data
            c = self.c.data if self.c is not None else 0.
        else:
            A = self.A
            B = self.B
            c = self.c if self.c is not None else 0.

        x_dim, u_dim = x.ndimension(), u.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if u_dim == 1:
            u = u.unsqueeze(0)

        z = x.mm(A.t()) + u.mm(B.t()) + c

        if x_dim == 1:
            z = z.squeeze(0)

        return z

    def grad_input(self, x, u):
        n_batch = x.size(0)
        A, B = self.A, self.B
        A = A.unsqueeze(0).repeat(n_batch, 1, 1)
        B = B.unsqueeze(0).repeat(n_batch, 1, 1)
        if not isinstance(x, Variable) and isinstance(A, Variable):
            A, B = A.data, B.data
        return A, B
