import torch
import pypose as pp
import matplotlib.pyplot as plt

class Simple2DNav(pp.module.System):
    """
    A simple 2D navigation model for testing MPPI on non-linear system
    """

    def __init__(self,dt,length=1.0):
        super().__init__()
        self._tau = dt
        self._length=length

    def state_transition(self, state, input, t=None):
        """
        The simple 2D nav has state: (x, y, theta) and input: (v, omega)
        """
        x, y, theta = state.moveaxis(-1, 0)
        v, omega = input.squeeze().moveaxis(-1, 0)
        xDot = v * torch.cos(theta)
        yDot = v * torch.sin(theta)

        thetaDot = omega.expand_as(xDot)
        _dstate = torch.stack((xDot, yDot, thetaDot), dim=-1)

        return (state.squeeze() + torch.mul(_dstate, self._tau)).unsqueeze(0)

    def observation(self, state, input, t=None):
        """
        Returns:
            [N x 3] Tensor of state (as the system is fully observable)
        """
        return state


if __name__ == '__main__':
    # Define initial state
    torch.manual_seed(0)
    x0 = torch.tensor([0., 0., 0.], requires_grad=False)
    dt=0.1
    target_position = torch.tensor([10., 10.])
    tolerance = 0.5
    cost_fn = lambda x, u, t: (x[..., 0] - 10)**2 + (x[..., 1] - 10)**2 + (u[..., 0])**2


    mppi = pp.module.MPPI(
        dynamics=Simple2DNav(dt),
        running_cost=cost_fn,
        nx=3,
        noise_sigma=torch.eye(2) * 1,
        num_samples=100,
        horizon=5,
        lambda_=0.01
        )

    N = 10
    X = [x0]
    U = []
    costs=[]
    i = 0
    xn=x0


    while torch.norm(xn[:2] - target_position) > tolerance:
        xc = X[-1]
        u, xn = mppi.forward(xc)
        print(u)
        print(xn)
        xn = xn[1]
        costs.append(cost_fn(xc, u[1], 1))
        X.append(xn)
        U.append(u)
        i += 1
        if i == 100:
            break


    # Convert all elements in X to 2D tensors
    X_2D = [x.unsqueeze(0) if x.dim() == 1 else x for x in X]

    # Now stack these tensors to make a single tensor for easier slicing
    X_tensor = torch.cat(X_2D, dim=0)

    # Extract x and y coordinates
    x_coords = X_tensor[:, 0]
    y_coords = X_tensor[:, 1]


    plt.figure(figsize=(10, 5))
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='b')
    plt.title('Trajectory of the System (X-Y Positions)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()
