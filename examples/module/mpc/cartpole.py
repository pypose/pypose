import os
import time
import torch
import argparse
import pypose as pp
import torch.optim as optim
import matplotlib.pyplot as plt


class CartPole(pp.module.NLS):
        def __init__(self, dt, length, cartmass, polemass, gravity):
            super().__init__()
            self.tau = dt
            self.length = length
            self.cartmass = cartmass
            self.polemass = polemass
            self.gravity = gravity
            self.poleml = self.polemass * self.length
            self.totalMass = self.cartmass + self.polemass

        def state_transition(self, state, input, t=None):
            x, xDot, theta, thetaDot = state.squeeze()
            force = input.squeeze()
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)

            temp = (force + self.poleml * thetaDot**2 * sintheta) / self.totalMass
            thetaAcc = (self.gravity * sintheta - costheta * temp) / \
                (self.length * (4.0 / 3.0 - self.polemass * costheta**2 / self.totalMass))
            xAcc = temp - self.poleml * thetaAcc * costheta / self.totalMass
            _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc))
            return (state.squeeze() + torch.mul(_dstate, self.tau)).unsqueeze(0)

        def observation(self, state, input, t=None):
            return state


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MPC Nonl-inear Learning Example \
                                     (Cartpole)')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/mpc/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    n_batch, n_state, n_ctrl, T = 1, 4, 1, 5
    dt = 0.01
    g = 9.81
    time  = torch.arange(0, T, device=args.device) * dt
    current_u = torch.sin(time).unsqueeze(1).unsqueeze(0)

    # expert
    exp = dict(
        Q = torch.tile(torch.eye(n_state + n_ctrl, device=args.device), \
                       (n_batch, T, 1, 1)),
        p = torch.tensor([[[ 0.1945,  0.2579,  0.1655,  1.0841, -0.6235],
                           [-0.3816, -0.4376, -0.7798, -0.3692,  1.4181],
                           [-0.4025,  1.0821, -0.0679,  0.5890,  1.1151],
                           [-0.0610,  0.0656, -1.0557,  0.8769, -0.5928],
                           [ 0.0123,  0.3731, -0.2426, -1.5464,  0.0056]]],
                           device=args.device),
        len = torch.tensor(1.5).to(args.device),
        m_cart = torch.tensor(20.0).to(args.device),
        m_pole = torch.tensor(10.0).to(args.device))

    torch.manual_seed(0)
    len = torch.tensor(2.0).to(args.device).requires_grad_()
    m_pole = torch.tensor(11.1).to(args.device).requires_grad_()

    def get_loss(x_init, _len, _m_pole):

        # expert
        solver_exp = CartPole(dt, exp['len'], exp['m_cart'], exp['m_pole'], g)
        stepper_exp = pp.utils.ReduceToBason(steps=15, verbose=False)
        mpc_expert = pp.module.MPC(solver_exp, exp['Q'], exp['p'], T, stepper=stepper_exp)
        x_true, u_true, cost_true = mpc_expert(dt, x_init, u_init=current_u)

        #agent
        solver_agt = CartPole(dt, _len, exp['m_cart'], _m_pole, g)
        stepper_agt = pp.utils.ReduceToBason(steps=15, verbose=False)
        mpc_agt = pp.module.MPC(solver_agt, exp['Q'], exp['p'], T, stepper=stepper_agt)
        x_pred, u_pred, cost_pred = mpc_agt(dt, x_init, u_init=current_u)

        traj_loss = ((u_true - u_pred)**2).mean() + ((x_true - x_pred)**2).mean()

        return traj_loss

    opt = optim.RMSprop([len, m_pole], lr=1e-2)

    steps = 500
    traj_losses = []
    model_losses = []

    for i in range(steps):
        x_init = torch.tensor([[0, 0, torch.pi + torch.deg2rad(5.0*torch.randn(1)), 0]],
                              device=args.device)
        traj_loss = get_loss(x_init, len, m_pole)
        opt.zero_grad()
        traj_loss.backward()
        opt.step()

        model_loss = torch.mean((len - exp['len'])**2) + \
                    torch.mean((m_pole - exp['m_pole'])**2)

        traj_losses.append(traj_loss.item())
        model_losses.append(model_loss.item())

        plot_interval = 1
        if i % plot_interval == 0:
            os.system('./plot.py "{}" &'.format(args.save))
            print("Length of pole of the agent system = ", len)
            print("Length of pole of the exp system = ", exp['len'])
            print("Mass of pole of the agent system = ", m_pole)
            print("Mass of pole of the exp system = ", exp['m_pole'])
            print('{:04d}: traj_loss: {:.8f} model_loss: {:.8f}'.format(
            i, traj_loss.item(), model_loss.item()))

    plt.subplot(2, 1, 1)
    plt.plot(range(steps), traj_losses)
    plt.ylabel('Trajectory Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(steps), model_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Model Loss')

    figure = os.path.join(args.save + 'cartpole.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
