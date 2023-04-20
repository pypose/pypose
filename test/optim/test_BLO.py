import pypose as pp
import torch as torch
from torch import matmul as mult

class LQR_Solver(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B, C, c, T, x0):
        # Set number of inputs and outputs
        n_state = self.A.size(0)
        n_input = self.B.size(1)
        n_all = n_state + n_input

        # Construct cost matrix
        cost_mat = torch.block_diag(*self.C)

        # Construct dynamics solver matrix
        AB = torch.concat((self.A, self.B), dim = 1).unsqueeze(0).repeat(self.T - 1, 1, 1)
        AB = torch.block_diag(*AB)
        dynamics = torch.zeros(n_state * self.T, n_all * self.T)
        dynamics[:n_state,:n_state] = torch.eye(n_state)
        dynamics[n_state:n_state + AB.size(0), :AB.size(1)] = AB
        idx1c = torch.linspace(n_all, n_all * (T - 1), T - 1, dtype=int)
        idx1r = n_state * torch.linspace(1, T - 1, T - 1, dtype=int)
        for i in range(0,n_state):
            dynamics[idx1r + i, idx1c + i] = -1

        # Create full matrix
        zero_mat = torch.zeros(dynamics.size(0), dynamics.size(0))
        full_mat = torch.cat((torch.cat((cost_mat, dynamics.transpose(0, 1)), dim = 1),
                              torch.cat((dynamics, zero_mat),                 dim = 1)), dim = 0)

        # Create solution vector
        sol = torch.zeros(self.A.size(0) * T)
        sol[:n_state] = x0
        sol = torch.cat((torch.zeros(cost_mat.size(0)), sol), dim = 0).unsqueeze(-1)

        # Final solution
        tau_mu = torch.linalg.solve(full_mat, sol).squeeze()

        tau_star = tau_mu[:dynamics.size(1)]
        mu_star = tau_mu[dynamics.size(1):]

        return tau_star, mu_star

class InnerCostAndConstraints(pp.optim.InnerModel):
    def __init__(self, A, B, C, c, T) -> None:
        super().__init__(self)

        A = torch.nn.Parameter(A.clone(), requires_grad = True)
        B = torch.nn.Parameter(B.clone(), requires_grad = True)

        self.C = C
        self.c = c
        self.T = T

        self.x0 = None

    def forward(self, input):
        # Calculate cost
        cost = 0.5 * mult(mult(input, torch.block_diag((self.C)), input)) \
               + torch.dot(input, self.c.flatten())

        # Calculate constraints
        n_state = self.A.size(0)
        n_input = self.B.size(1)
        n_all = n_state + n_input

        AB = torch.concat((self.A, self.B), dim = 1).unqsueeze(0).repeat(self.T - 1, 1, 1)
        AB = torch.block_diag(*AB)
        dynamics = torch.zeros(n_state * self.T, n_all * self.T)
        dynamics[:n_state, :n_state] = torch.eye(n_state)
        dynamics[n_state: n_state + AB.size(0), :AB.size(1)] = AB
        idx1c = torch.linspace(n_all, n_all * (self.T - 1), self.T - 1, dtype = int)
        idx1r = n_state * torch.linspace(1, self.T - 1, self.T - 1, dtype = int)
        for i in range(0, n_state):
            dynamics[idx1r + i, idx1c + i] = -1

        b = torch.zeros(dynamics.size(0))
        b[:n_state] = self.x0

        return mult(dynamics, input) - b

class OuterLoss(pp.optim.InnerModel):
    def __init__(self, tau_hat, inner_model):
        super.__init__(self, inner_model)
        self.tau_hat = tau_hat

    def forward(self, tau_star):
        return torch.sqrt(torch.sum((self.tau_hat - tau_star) ** 2))

if __name__ == "__main__":

    torch.manual_seed(6)

    n_state = 3
    n_ctrl = 3
    n_all = n_state + n_ctrl
    alpha = 0.2
    T = 5

    C = torch.squeeze(torch.randn(T, 1, n_all, n_all))
    C = torch.matmul(C.mT, C)
    c = torch.squeeze(torch.randn(T, 1, n_all))

    expert = dict(
        C = C,
        c = c,
        A = (torch.eye(n_state) + alpha*torch.randn(n_state, n_state)),
        B = torch.randn(n_state, n_ctrl)
    )

    A = torch.eye(n_state) + alpha*torch.randn(n_state, n_state)
    B = torch.randn(n_state, n_ctrl)
    x0 = torch.randn(n_state)

    lqrLearn = pp.optim.BLO(OuterLoss, LQR_Solver)

    for iter in range(0, 5000):
        x0 = torch.randn(n_state)

        OuterLoss.tau_hat = LQR_Solver(dict['A'], dict['B'], dict['C'], dict['c'], T, x0)
        lqrLearn.step(x0)
