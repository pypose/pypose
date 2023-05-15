import math
import pypose as pp
import torch as torch
import matplotlib.pyplot as plt

from pypose.module.mppi import MPPI

# The class
class CartPole(pp.module.System):
    """
    Cartpole dynamics for testing MPPI on non-linear system
    """
    def __init__(self):
        super(CartPole, self).__init__()
        self._tau = 0.01
        self._length = 1.5
        self._cartmass = 20.0
        self._polemass = 0.1
        self._gravity = 9.81
        self._polemassLength = self._polemass * self._length
        self._totalMass = self._cartmass + self._polemass

    def state_transition(self, state, input, t=None):
        """
        Args:
            state: [N x 4] Tensor of states
            inputs: [N x 2] Tensor of controls
            t: Tensor of time (not used as CartPole is time-invariant)
        Returns:
            [N x 4] Tensor of time derivatives for integration
        """
        x, xDot, theta, thetaDot = state.moveaxis(-1, 0)
        force = input.squeeze().moveaxis(-1, 0)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (
            force + self._polemassLength * thetaDot**2 * sintheta
        ) / self._totalMass
        thetaAcc = (self._gravity * sintheta - costheta * temp) / (
            self._length * (4.0 / 3.0 - self._polemass * costheta**2 / self._totalMass)
        )
        xAcc = temp - self._polemassLength * thetaAcc * costheta / self._totalMass

        _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc), dim=-1)

        return state + torch.mul(_dstate, self._tau)

    def observation(self, state, input, t=None):
        """
        Returns:
            [N x 4] Tensor of state (as the system is fully observable)
        """
        return state

class CartPole2(pp.module.System):
    def __init__(self):
        """
        In addition to the regular CartPole, also allow for varying the length of the pole.
        """
        super(CartPole2, self).__init__()
        self._tau = 0.01
        self._cartmass = torch.tensor(20.0, requires_grad=False)
        self._polemass = torch.tensor(0.1, requires_grad=False)
        self._gravity = torch.tensor(9.81, requires_grad=False)

    def set_params(self, params):
        self._length = params.item()

    def state_transition(self, state, input, params, t=None):
        """
        Args:
            state: [N x 4] Tensor of states
            inputs: [N x 2] Tensor of controls
            params: [N x 1] Tensor of system params (in this case, pole length)
            t: Tensor of time (not used as CartPole is time-invariant)
        Returns:
            [N x 4] Tensor of time derivatives for integration
        """
        x, xDot, theta, thetaDot = state.moveaxis(-1, 0)
        force = input.squeeze().moveaxis(-1, 0)
        length = params.squeeze().moveaxis(-1, 0)
        polemassLength = self._polemass * length
        totalMass = self._cartmass + self._polemass

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (
            force + polemassLength * thetaDot**2 * sintheta
        ) / totalMass
        thetaAcc = (self._gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - self._polemass * costheta**2 / totalMass)
        )
        xAcc = temp - polemassLength * thetaAcc * costheta / totalMass

        _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc), dim=-1)

        return state + torch.mul(_dstate, self._tau)

    def observation(self, state, input, t=None):
        """
        Returns:
            [N x 4] Tensor of state (as the system is fully observable)
        """
        return state

    def parameters(self):
        """
        Returns:
            [N x 4] Tensor of system parameters
        """
        return [
            self._length,
            self._cartmass,
            self._polemass,
            self._gravity
        ]

def visualize(system, traj, controls):
    """
    pyplot visualization of CartPole for debugging
    Args:
        system: The CartPole system (to pull system params from)
        traj: [T x 4] Tensor of states to plot
        controls: [T x 2] Tensor of controls to plot
    Returns:
        None
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.show(block=False)

    l = system._length
    x = traj[:, 0]
    th = traj[:, 2]

    #plot cartpole
    for i in range(traj.shape[0]):
        for ax in axs:
            ax.cla()

        axs[0].set_xlim(x[i] - 3., x[i] + 3.)
        axs[0].set_ylim(-3., 3.)

        axs[0].plot(x, torch.zeros_like(x), alpha=0.3, c='r')
        axs[0].plot(x + l*torch.sin(th), l*torch.cos(th), alpha=0.3, c='k')

        axs[0].scatter(x[i], 0., c='r', label='com')
        axs[0].scatter(x[i] + l*torch.sin(th[i]), l*torch.cos(th[i]), c='k', label='endpt')
        
        #plot states
        for j, (label, color) in enumerate(zip(['x', 'xd', 'th', 'thd'], ['r', 'g', 'b', 'y'])):
            axs[1].plot(traj[:, j], label=label, c=color)
            axs[1].scatter(i, traj[i, j], marker='.', c=color)

        #plot control
        for j, label in enumerate(['fx']):
            axs[2].plot(controls[:, j], label=label, c='r')
            axs[2].scatter(i, controls[i, j], c='r', marker='.')

        for ax in axs:
            ax.legend()

        plt.pause(system._tau)

if __name__ == '__main__':
    """
    This script tests the ability of MPPI to optimize non-linear dynamics (CartPole)
    
    In addition to optimizing the cost function, also use gradients to identify the length of the pole.
    """
    import math
    cartpole = CartPole()
    cartpole._tau = 0.1

    cartpole_learner = CartPole2()
    cartpole_learner._tau = 0.1

    ## test viz and simple traj ##
    dt = 0.1
    N = 200
    time = torch.arange(N).unsqueeze(-1) * dt
    U = torch.sin(time) * 0.
    x0 = torch.tensor([0., 0.0, 0.0, 0.0], requires_grad=False)
    X = [x0]
    for u in U:
        xc = X[-1].clone()
        xn = cartpole.forward(xc, u)[0]
        X.append(xn)

    X = torch.stack(X, dim=0)

#    visualize(cartpole, X, U)

#    cost_fn = lambda x, u: (x[..., 1]) * 100.0 + (x[..., 1].pow(2)) * 100. + (x[..., 3].pow(2)) * 100.0

    cost_fn = lambda x, u: (x[..., 1]) * 0.0 + (x[..., 1].pow(2)) * 0. + (x[..., 3].pow(2)) * 0.0

    mppi = MPPI(
#        dynamics=cartpole_learner,
        dynamics=cartpole,
        running_cost=cost_fn,
        nx=4,
        noise_sigma=torch.eye(1) * 1.0,
        num_samples=5000,
        horizon=100,
        lambda_=0.01,
        u_scale=1.0
    )


    # roll out an initial trajectory with ground-truth dynamics
    X = [x0]
    U = []
    params = torch.tensor([1.5, 20.0, 0.1, 9.81], requires_grad=False)
    for i in range(N):
        xc = X[-1]
        u = mppi.command(xc)
        xn = cartpole.forward(xc, u)[0]
        X.append(xn)
        U.append(u)

    ## dynamics update ##
#    mppi_u = mppi.perturbed_action[mppi.cost_total.argmin()]
#    gt_X = [xc]
#    for _u in mppi_u:
#        gt_X.append(cartpole.forward(gt_X[-1], _u)[0])

#    gt_X = torch.stack(gt_X[1:], dim=0).detach()
#    mppi_X = mppi.states[0, mppi.cost_total.argmin()]

    
#    mppi_X = torch.stack(X, dim=0)

    X = torch.stack(X, dim=0).detach()
    U = torch.stack(U, dim=0).detach()

    params = torch.tensor([2.0], requires_grad=True)
    opt = torch.optim.RMSprop([params], lr=0.01)

    # perform GD to solve for cartpole length
    for i in range(100):
        with torch.autograd.set_detect_anomaly(True):
            mppi_X = [x0.clone()]
            gt_X = [x0.clone()]
            for _u in U:
                mppi_X.append(cartpole_learner.state_transition(mppi_X[-1], _u, params))
                gt_X.append(cartpole.forward(gt_X[-1], _u)[0])

            gt_X = torch.stack(gt_X, dim=0).detach()
            mppi_X = torch.stack(mppi_X, dim=0)

            loss = (gt_X - mppi_X).pow(2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

        print('Loss = {:.4f}, Cartpole params: {}'.format(loss.detach(), params))

    print('final cost: {:.2f}'.format(cost_fn(X, U).sum()))
    cartpole_learner.set_params(params.detach())
    visualize(cartpole_learner, X[1:].detach(), U.detach())
