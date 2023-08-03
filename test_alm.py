import torch
import time
import numpy as np
import pypose as pp
from torch import nn
import numpy as np
from pypose.optim import ALM
from torch import matmul as mult

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
        dynamics = torch.zeros(n_state * T, n_all * T)
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
        sol = torch.zeros(A.size(0) * T)
        sol[:n_state] = x0
        sol = torch.cat((torch.zeros(cost_mat.size(0)), sol), dim = 0).unsqueeze(-1)

        # Final solution
        tau_mu = torch.linalg.solve(full_mat, sol).squeeze()

        tau_star = tau_mu[:dynamics.size(1)]
        mu_star = tau_mu[dynamics.size(1):]

        return tau_star, mu_star

class Timer:
    def __init__(self):
        self.synchronize()
        self.start_time = time.time()
    
    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  

    def tic(self):
        self.start()

    def show(self, prefix="", output=True):
        self.synchronize()
        duration = time.time()-self.start_time
        if output:
            print(prefix+"%fs" % duration)
        return duration

    def toc(self, prefix=""):
        self.end()
        print(prefix+"%fs = %fHz" % (self.duration, 1/self.duration))
        return self.duration

    def start(self):
        self.synchronize()
        self.start_time = time.time()

    def end(self, reset=True):
        self.synchronize()
        self.duration = time.time()-self.start_time
        if reset:
            self.start_time = time.time()
        return self.duration

def test_tensor_complex():
    class PoseNet(nn.Module):
        def __init__(self, T, C, c, n_all):
            super().__init__()
            self.x = torch.nn.Parameter(torch.randn(T, n_all).flatten())
            self.C = C
            self.c = c

        def forward(self, input=None):
            cost = 0.5 * mult(mult(self.x, torch.block_diag(*self.C)), self.x)
            return cost

    class ConstrainNet(nn.Module):
        def __init__(self, A, B, T, net,x0):
            super().__init__()
            self.A = A
            self.B = B
            self.T = T
            self.net = net
            self.x0 = x0
            
        def forward(self, input=None):
            n_state = self.A.size(0)
            n_input = self.B.size(1)
            n_all = n_state + n_input

            AB = torch.concat((self.A, self.B), dim = 1).unsqueeze(0).repeat(self.T - 1, 1, 1)
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

            return mult(dynamics, self.net.x) - b
    device = torch.device("cpu")
    torch.manual_seed(6)
    n_state = 3
    n_ctrl = 3
    n_all = n_state + n_ctrl
    alpha = 0.2
    T = 5
    
    C = torch.squeeze(torch.randn(T, 1, n_all, n_all))
    C = torch.matmul(C.mT, C)
    c = torch.squeeze(torch.randn(T, 1, n_all))
    
    A = torch.eye(n_state) + alpha*torch.randn(n_state, n_state)
    B = torch.randn(n_state, n_ctrl)
    x0 = torch.randn(n_state)
    solver = LQR_Solver()
    posnet = PoseNet(T, C, c, n_all).to(device)
    constraints = ConstrainNet(A, B, T, posnet, x0).to(device)
    
    optimizer = ALM(model=posnet, constraints=constraints, object_decrease_tolerance=1e-7, inner_iter=300)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer.optim, step_size=20, gamma=0.5)
    timer = Timer()

    for idx in range(100):
        loss, lmd, = optimizer.step(input=None)
        # scheduler.step()
        if optimizer.terminate:
            break
    print('-----------optimized result----------------')
    print('Done', timer.toc())
    print('object f(x):', posnet())
    print('final violation:\n', torch.norm(constraints()))
    print("Lambda*:\n", lmd)
    print('tau*:', posnet.x)
    tau, mu = solver(A, B, C, T, x0)
    print('Lambda true:\n', mu)
    print('tau true:\n', tau)


def test_tensor():
    class PoseNet(nn.Module):
        def __init__(self, *dim):
            super().__init__()
            init = torch.randn(*dim)
            self.pose = torch.nn.Parameter(init) # torch([x1,x2])

        def forward(self, input):
            result = -self.pose.prod() # get tensor(-x1*x2)
            return result

    class ConstrainNet(nn.Module):
        def __init__(self, objective_net):
            super().__init__()
            self.net = objective_net
            
        def forward(self, input=None):
            constraint_violation = torch.square(torch.norm(self.net.pose, p=2)) - 2
            # constraint_violation = torch.square(self.net.pose[0]) + torch.square(self.net.pose[1]) + torch.square(self.net.pose[2]) - 2
            return constraint_violation.unsqueeze(0)
    device = torch.device("cpu")
    input = None
    posnet = PoseNet(5).to(device)
    constraints = ConstrainNet(posnet).to(device)
    optimizer = ALM(model=posnet, constraints=constraints, penalty_safeguard=1e3, inner_iter=400)
    
    
    for idx in range(20):
        loss, lmd, = optimizer.step(input)
        if optimizer.terminate:
            break
            
    print('-----------optimized result----------------')
    print("Lagrangian Multiplier Lambda:",lmd)
    print(posnet.pose)

def test_lietensor():
    class PoseNet(nn.Module):
        def __init__(self, *dim):
            super().__init__()
            # self.pose = pp.Parameter(pp.so3([[0, 0, 1.0]]))
            self.pose = pp.Parameter(pp.randn_so3(*dim))

        def forward(self, input):
            result = (self.pose.Exp() @ input).matrix() - torch.eye(3)
            return torch.norm(result)

    class ConstrainNet(nn.Module):
        def __init__(self, objective_net):
            super().__init__()
            self.net = objective_net
            
        def forward(self, input=None):
            fixed_euler_angles = np.array([[0.0, 0.0, 0.0]])
            fixed_quaternion = pp.euler2SO3(fixed_euler_angles)
            quaternion = self.net.pose.Exp()
            difference_quaternions = torch.sub(quaternion, fixed_quaternion)
            distance = torch.norm(difference_quaternions, p=2, dim=1)
            d_fixed = 0.35
            constraint_violation = distance - d_fixed
            return constraint_violation

    device = torch.device("cpu")
    euler_angles = np.array([[0.0, 0.0, np.pi/4]])  # [x, y, z]
    quaternion = pp.euler2SO3(euler_angles)
    quaternion = torch.tensor(quaternion, dtype=torch.float)
    input = pp.SO3(quaternion)
    
    posnet = PoseNet(1).to(device)
    initial_pose = posnet.pose.detach().clone()
    constraints = ConstrainNet(posnet).to(device)

    optimizer = ALM(model=posnet, constraints=constraints, inner_iter=200)
   

    for idx in range(20):
        loss, lmd, = optimizer.step(input)
        if optimizer.terminate:
            break
    print('-----------optimized result----------------')
    decimal_places = 4
    print("Lambda:",lmd)
    print('x axis:', np.around(posnet.pose.detach().numpy(), decimals=decimal_places))
    print('f(x):', posnet(input))
    print('final violation:', constraints())
    
if __name__ == "__main__":
    test_tensor()
    test_tensor_complex()
    test_lietensor()