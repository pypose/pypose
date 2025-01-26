from torch import nn
import torch, pypose as pp
from pypose.optim import SAL
from pypose.utils import Prepare
from pypose.optim.scheduler import StopOnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LQR_Solver(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, A, B, C, T, x0):
        n_state = A.size(0)
        n_input = B.size(1)
        n_all = n_state + n_input

        # Construct cost matrix
        cost_mat = torch.block_diag(*C)

        # Construct dynamics solver matrix
        AB = torch.concat((A, B), dim = 1).unsqueeze(0).repeat(T - 1, 1, 1)
        AB = torch.block_diag(*AB)

        dynamics = torch.zeros(n_state * T, n_all * T, device=device)

        dynamics[:n_state,:n_state] = torch.eye(n_state, device=device)
        dynamics[n_state:n_state + AB.size(0), :AB.size(1)] = AB
        idx1c = torch.linspace(n_all, n_all * (T - 1), T - 1, dtype=int)
        idx1r = n_state * torch.linspace(1, T - 1, T - 1, dtype=int)
        for i in range(0,n_state):
            dynamics[idx1r + i, idx1c + i] = -1

        # Create full matrix

        zero_mat = torch.zeros(dynamics.size(0), dynamics.size(0), device=device)

        full_mat = torch.cat((torch.cat((cost_mat, dynamics.transpose(0, 1)), dim = 1),
                              torch.cat((dynamics, zero_mat),                 dim = 1)), dim = 0)

        # Create solution vector

        sol = torch.zeros(A.size(0) * T, device=device)
        sol[:n_state] = x0
        sol = torch.cat((torch.zeros(cost_mat.size(0), device=device), sol), dim = 0).unsqueeze(-1)

        # Final solution
        tau_mu = torch.linalg.solve(full_mat, sol).squeeze()

        tau_star = tau_mu[:dynamics.size(1)]
        mu_star = tau_mu[dynamics.size(1):]

        return tau_star, mu_star


class AlmOptimExample:

    def tensor_complex(self):
        class TensorModel(nn.Module):
            def __init__(self, T, C, n_all, A, B, x0) -> None:
                super().__init__()
                self.x = nn.Parameter(torch.randn(T, n_all, device=device).flatten())
                self.C = C
                self.A = A
                self.B = B
                self.x0 = x0
                self.T = T

            def objective(self, inputs):

                self.C = self.C

                cost = 0.5 * self.x @ torch.block_diag(*self.C) @ self.x
                return cost

            def constrain(self, inputs):
                n_state = self.A.size(0)
                n_input = self.B.size(1)
                n_all = n_state + n_input

                AB = torch.concat((self.A, self.B), dim = 1).unsqueeze(0).repeat(self.T - 1, 1, 1)
                AB = torch.block_diag(*AB)

                dynamics = torch.zeros(n_state * self.T, n_all * self.T, device=self.x.device)

                dynamics[:n_state, :n_state] = torch.eye(n_state)
                dynamics[n_state: n_state + AB.size(0), :AB.size(1)] = AB
                idx1c = torch.linspace(n_all, n_all * (self.T - 1), self.T - 1, dtype = int)
                idx1r = n_state * torch.linspace(1, self.T - 1, self.T - 1, dtype = int)
                for i in range(0, n_state):
                    dynamics[idx1r + i, idx1c + i] = -1


                b = torch.zeros(dynamics.size(0), device=self.x.device)

                b[:n_state] = self.x0

                return dynamics @ self.x - b

            def forward(self, inputs):
                return self.objective(inputs), self.constrain(inputs)

        torch.manual_seed(6)
        n_state = 3
        n_ctrl = 3
        n_all = n_state + n_ctrl
        alpha = 0.2
        T = 5

        C = torch.squeeze(torch.randn(T, 1, n_all, n_all, device=device))
        C = torch.matmul(C.mT, C)

        A = torch.eye(n_state, device=device) + alpha * torch.randn(n_state, n_state, device=device)
        B = torch.randn(n_state, n_ctrl, device=device)
        x0 = torch.randn(n_state, device=device)


        model = TensorModel(T, C, n_all, A, B, x0)
        input = None

        outopt = SAL(model=model, penalty=1, shield=1e3, scale=2, hedge=0.9)
        outsch = StopOnPlateau(outopt, steps=120, patience=10, decreasing=1e-6, verbose=True)

        inopt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        outopt.inner_scheduler(Prepare(ReduceLROnPlateau, optimizer=inopt, min_lr=1e-2), steps=200)

        while outsch.continual():
            loss = outopt.step(input)
            outsch.step(loss)

        print("Lambda*:\n", outopt.model.lmd)
        print('tau*:', model.x)
        solver = LQR_Solver()
        tau, mu = solver(A, B, C, T, x0)
        print('Lambda true:\n', mu)
        print('tau true:\n', tau)

    def lietensor(self):
        class PoseInvConstrained(nn.Module):
            def __init__(self, *dim) -> None:
                super().__init__()
                self.pose = pp.Parameter(pp.randn_so3(*dim, device=device))

            def objective(self, inputs):

                result = (self.pose.Exp() @ inputs).matrix() - torch.eye(3, device=device)

                return torch.norm(result)

            def constrain(self, inputs):
                fixed_euler_angles = torch.Tensor([[0.0, 0.0, 0.0]], device=device)

                fixed_quaternion = pp.euler2SO3(fixed_euler_angles)

                quaternion = self.pose.Exp()
                difference_quaternions = torch.sub(quaternion, fixed_quaternion)
                distance = torch.norm(difference_quaternions, p=2, dim=1)
                d_fixed = 0.35
                constraint_violation = distance - d_fixed
                return constraint_violation

            def forward(self, inputs):
                return self.objective(inputs), self.constrain(inputs)

        euler_angles = torch.Tensor([[0.0, 0.0, torch.pi/4]], device=device)
        quaternion = pp.euler2SO3(euler_angles)
        input = pp.SO3(quaternion)

        model = PoseInvConstrained(1)


        outopt = SAL(model, penalty=1, shield=1e3, scale=2, hedge=0.9)
        outsch = StopOnPlateau(outopt, steps=20, patience=5, decreasing=1e-6, verbose=True)

        inopt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
        # insch = ReduceLROnPlateau(optimizer=inopt, min_lr=1e-2, verbose=True)
        outopt.inner_scheduler(Prepare(ReduceLROnPlateau, optimizer=inopt, min_lr=1e-2, verbose=True), steps=200)

        while outsch.continual():
            loss = outopt.step(input)
            outsch.step(loss)

        print("Lambda:", outopt.model.lmd)
        print('x axis:', model.pose.cpu().detach().numpy())

        print('f(x):', model.objective(input))
        print('final violation:', model.constrain(input))

if __name__ == "__main__":
    alm = AlmOptimExample()
    alm.lietensor()
    alm.tensor_complex()
