import os
import torch
import shutil
import argparse
import setproctitle
import pypose as pp
import pickle as pkl
import torch.optim as optim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_batch', type=int, default=1)
    parser.add_argument('--n_state', type=int, default=4)
    parser.add_argument('--n_ctrl', type=int, default=1)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work2', type=str, default='work2')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = '.'.join(["{}={}".format(x, getattr(args, x))
                  for x in ['n_batch', 'n_state', 'n_ctrl', 'T']])
    setproctitle.setproctitle('bamos.lqr.'+t+'.{}'.format(args.seed))
    if args.save is None:
        args.save = os.path.join(args.work2, t, str(args.seed))
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    n_batch, n_state, n_ctrl, T= args.n_batch, args.n_state, args.n_ctrl, args.T
    n_sc = n_state + n_ctrl

    class CartPole(pp.module.NLS):
        def __init__(self, dt, length, cartmass, polemass, gravity):
            super().__init__()
            self.tau = dt
            self.length = length
            self.cartmass = cartmass
            self.polemass = polemass
            self.gravity = gravity
            self.polemassLength = self.polemass * self.length
            self.totalMass = self.cartmass + self.polemass

        def state_transition(self, state, input, t=None):
            x, xDot, theta, thetaDot = state.squeeze()
            force = input.squeeze()
            costheta = torch.cos(theta)
            sintheta = torch.sin(theta)

            temp = (
                force + self.polemassLength * thetaDot**2 * sintheta
            ) / self.totalMass
            thetaAcc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.polemass * costheta**2 / self.totalMass)
            )
            xAcc = temp - self.polemassLength * thetaAcc * costheta / self.totalMass

            _dstate = torch.stack((xDot, xAcc, thetaDot, thetaAcc))

            return (state.squeeze() + torch.mul(_dstate, self.tau)).unsqueeze(0)

        def observation(self, state, input, t=None):
            return state

    expert_seed = 7
    assert expert_seed != args.seed
    torch.manual_seed(expert_seed)

    Q = torch.tile(torch.eye(n_state + n_ctrl, device=device), (n_batch, T, 1, 1))
    p = torch.randn(n_batch, T, n_sc)
    dt = 0.01
    g = 9.81
    time  = torch.arange(0, T, device=device) * dt

    expert = dict(
        Q = torch.tile(torch.eye(n_state + n_ctrl, device=device), (n_batch, T, 1, 1)),
        p = torch.randn(n_batch, T, n_sc).to(device),
        len = torch.tensor(1.5).to(device),
        m_cart = torch.tensor(20).to(device),
        m_pole = torch.tensor(10).to(device),
    )
    fname = os.path.join(args.save, 'expert.pkl')
    with open(fname, 'wb') as fi:
        pkl.dump(expert, fi)

    current_u = torch.sin(time).unsqueeze(1).unsqueeze(0)
    x_init = torch.tensor([[0, 0, torch.pi, 0]], device=device)
    expert_cartPoleSolver = CartPole(dt, expert['len'], expert['m_cart'], expert['m_pole'], g).to(device)

    torch.manual_seed(args.seed)
    len = torch.tensor(1.55).to(device).requires_grad_()
    m_cart = torch.tensor(20.3).to(device).requires_grad_()
    m_pole = torch.tensor(10.4).to(device).requires_grad_()

    fname = os.path.join(args.save, 'cartpole losses.csv')
    loss_f = open(fname, 'w')
    loss_f.write('im_loss,mse\n')
    loss_f.flush()

    mpc_expert = pp.module.MPC(expert_cartPoleSolver, T, step=15).to(device)
    x_true, u_true, cost_true = mpc_expert(expert['Q'], expert['p'], x_init, dt, current_u)

    def get_loss(_len, _m_cart, _m_pole):
        x_init = torch.tensor([[0, 0, torch.pi, 0]], device=device)
        agent_cartPoleSolver = CartPole(dt, _len, _m_cart, _m_pole, g).to(device)
        mpc_agent = pp.module.MPC(agent_cartPoleSolver, T, step=15).to(device)
        x_pred, u_pred, cost_pred = mpc_agent(expert['Q'], expert['p'], x_init, dt, current_u)

        traj_loss = torch.mean((u_true - u_pred)**2) \
            + torch.mean((x_true - x_pred)**2)

        return traj_loss

    opt = optim.RMSprop((len, m_cart, m_pole), lr=1e-2)

    for i in range(1000):
        traj_loss = get_loss(len, m_cart, m_pole)
        opt.zero_grad()
        traj_loss.backward()
        opt.step()

        model_loss = torch.mean((len - expert['len'])**2) + \
                     torch.mean((m_cart - expert['m_cart'])**2) + \
                     torch.mean((m_pole - expert['m_pole'])**2)

        loss_f.write('{},{}\n'.format(traj_loss.item(), model_loss.item()))
        loss_f.flush()

        plot_interval = 1
        if i % plot_interval == 0:
            os.system('./plot.py "{}" &'.format(args.save))
            print("Length of pole of the agent system = ", len)
            print("Length of pole of the expert system = ", expert['len'])
            print("Mass of cart of the agent system = ", m_cart)
            print("Mass of cart of the expert system = ", expert['m_cart'])
            print("Mass of pole of the agent system = ", m_pole)
            print("Mass of pole of the expert system = ", expert['m_pole'])
            print('{:04d}: traj_loss: {:.8f} model_loss: {:.8f}'.format(
            i, traj_loss.item(), model_loss.item()))


if __name__=='__main__':
    main()
