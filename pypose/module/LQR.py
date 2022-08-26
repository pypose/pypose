import torch as torch

class DP_LQR:

    def __init__(self, n_state, n_ctrl, T, current_x=None, current_u=None):
        
        self.n_state = n_state
        self.n_ctrl = n_ctrl
        self.T = T
        self.current_x = current_x
        self.current_u = current_u

    def DP_LQR_backward(self, C, c, F, f):

        Ks = []
        ks = []
        Vtp1 = vtp1 = None
        for t in range(self.T-1, -1, -1): #range(start, stop, step): the stop number itself is always omitted.
            if t == self.T-1:
                Qt = C[t]
                qt = c[t]
            else:
                Ft = F[t]
                Ft_T = Ft.transpose(1,2)
                Qt = C[t] + Ft_T.matmul(Vtp1).matmul(Ft)
                if f is None or f.numel() == 0: #alias for nelement(): count the number of elements of the tensor
                    qt = c[t] + Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2) #Tensor multiplied Nonetype
                else:
                    ft = f[t]
                    qt = c[t] + Ft_T.bmm(Vtp1).bmm(ft.unsqueeze(2)).squeeze(2) + \
                        Ft_T.bmm(vtp1.unsqueeze(2)).squeeze(2)
            
            n_state = self.n_state
            Qt_xx = Qt[:, :n_state, :n_state]
            Qt_xu = Qt[:, :n_state, n_state:]
            Qt_ux = Qt[:, n_state:, :n_state]
            Qt_uu = Qt[:, n_state:, n_state:]
            qt_x = qt[:, :n_state]
            qt_u = qt[:, n_state:]

            if self.n_ctrl == 1:
                Kt = -(1./Qt_uu)*Qt_ux
                kt = -(1./Qt_uu.squeeze(2))*qt_u
            else:
                Qt_uu_inv = [torch.linalg.pinv(Qt_uu[i]) for i in range(Qt_uu.shape[0])]
                Qt_uu_inv = torch.stack(Qt_uu_inv) #Concatenates a sequence of tensors along a new dimension. All tensors need to be of the same size.
                Kt = -Qt_uu_inv.matmul(Qt_ux)
                kt = -Qt_uu_inv.bmm(qt_u.unsqueeze(2)).squeeze(2)

            Kt_T = Kt.transpose(1,2)

            Ks.append(Kt)
            ks.append(kt)

            Vtp1 = Qt_xx + Qt_xu.bmm(Kt) + Kt_T.bmm(Qt_ux) + Kt_T.bmm(Qt_uu).bmm(Kt)
            vtp1 = qt_x + Qt_xu.bmm(kt.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(qt_u.unsqueeze(2)).squeeze(2) + \
                Kt_T.bmm(Qt_uu).bmm(kt.unsqueeze(2)).squeeze(2)

        return Ks, ks

    def DP_LQR_forward(self, x_init, C, c, F, f, Ks, ks):
        x = self.current_x
        u = self.current_u
        #current_cost = None

        new_u = []
        new_x = [x_init]
        objs = []
        tau = []

        for t in range(self.T):
            t_rev = self.T-1-t
            Kt = Ks[t_rev]
            kt = ks[t_rev]
            new_xt = new_x[t]
            new_ut = Kt.bmm(new_xt.unsqueeze(2)).squeeze(2) + kt
            new_u.append(new_ut)

            new_xut = torch.cat((new_xt, new_ut), dim=1)
            if t < self.T-1:
                new_xtp1 = F[t].bmm(new_xut.unsqueeze(2)).squeeze(2)
                if f is not None and f.numel() > 0:
                    new_xtp1 += f[t]

                new_x.append(new_xtp1)

            #new_xut_T = new_xut.mT

            obj = 0.5*new_xut.unsqueeze(1).bmm(C[t]).bmm(new_xut.unsqueeze(2)).squeeze(1).squeeze(1) + torch.bmm(new_xut.unsqueeze(1), c[t].unsqueeze(2)).squeeze(1).squeeze(1)
            objs.append(obj)

            tau.append(new_xut)

        objs = torch.stack(objs)
            #current_cost = torch.sum(objs, dim=0)

        new_u = torch.stack(new_u)
        new_x = torch.stack(new_x)

        return new_x, new_u, objs, tau #,current_X


    def DP_LQR_costates(self, tau, C, c, F):

        lambda_dual = []

        for t in range(self.T-1, -1, -1):
            t_rev_new = self.T-1-t
            tau_t = tau[t_rev_new]
            Ct = C[t]
            ct = c[t]
            Ft = F[t]
            n_state = self.n_state
            Ct_x = Ct[:, :n_state, :]
            ct_x = ct[:, :n_state]

            if t == self.T-1:
                lambda_final = lambda_tp1 = Ct_x.bmm(tau_t.unsqueeze(2)).squeeze(2) + ct_x
            else: 
                Ft_T = Ft.transpose(1,2)
                Ft_x_T = Ft_T[:, :n_state, :]
                lambda_t = Ft_x_T .bmm(lambda_tp1.unsqueeze(2)).squeeze(2) + Ct_x.bmm(tau_t.unsqueeze(2)).squeeze(2) + ct_x
                lambda_tp1 = lambda_t
                lambda_dual.append(lambda_t)
        
        lambda_dual.reverse()
        lambda_dual.append(lambda_final)
        lambda_dual = torch.stack(lambda_dual)

        return lambda_dual