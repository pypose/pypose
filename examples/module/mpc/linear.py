import os
import time
import torch
import shutil
import argparse
import setproctitle
import pypose as pp
import pickle as pkl
import torch.optim as optim

class TrainMPC:
    def main(self, device='cpu'):

        parser = argparse.ArgumentParser()
        parser.add_argument('--n_batch', type=int, default=5)
        parser.add_argument('--n_state', type=int, default=3)
        parser.add_argument('--n_ctrl', type=int, default=3)
        parser.add_argument('--T', type=int, default=5)
        parser.add_argument('--save', type=str)
        parser.add_argument('--work', type=str, default='work')
        parser.add_argument('--no-cuda', action='store_true')
        parser.add_argument('--seed', type=int, default=0)
        args = parser.parse_args()

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        t = '.'.join(["{}={}".format(x, getattr(args, x))
                    for x in ['n_batch', 'n_state', 'n_ctrl', 'T']])
        setproctitle.setproctitle('bamos.lqr.'+t+'.{}'.format(args.seed))
        if args.save is None:
            args.save = os.path.join(args.work, t, str(args.seed))
        if os.path.exists(args.save):
            shutil.rmtree(args.save)
        os.makedirs(args.save, exist_ok=True)

        device = torch.device("cuda" if args.cuda else "cpu")
        n_batch, n_state, n_ctrl, T= args.n_batch, args.n_state, args.n_ctrl, args.T
        n_sc = n_state + n_ctrl

        C = torch.eye(n_state, device=device)
        D = torch.zeros(n_state, n_ctrl, device=device)
        c1 = torch.zeros(n_state, device=device)
        c2 = torch.zeros(n_state, device=device)
        dt = 1

        expert = dict(
            Q = torch.tile(torch.eye(n_sc, device=device), (n_batch, T, 1, 1)),
            p = torch.tile(torch.tensor([ 0.6336, -0.2203, -0.1395, -0.7664,  0.8874,  0.8153], \
                                        device=device), (n_batch, T, 1)),
            A = torch.tensor([[ 1.1267, -0.0441, -0.0279],
                              [-0.1533,  1.1775,  0.1631],
                              [ 0.1618,  0.1238,  0.9489]], device=device),
            B = torch.tensor([[ 0.4567,  0.7805,  0.0319],
                              [-0.5938, -0.5724,  0.0422],
                              [-0.1804, -0.2535,  1.7218]], device=device),
        )
        fname = os.path.join(args.save, 'expert.pkl')
        with open(fname, 'wb') as fi:
            pkl.dump(expert, fi)

        torch.manual_seed(args.seed)
        A = torch.tensor([[ 1.2082, -0.1587, -0.3358],
                          [ 0.2137,  0.8831, -0.1797],
                          [ 0.1807,  0.2676,  0.7561]], device=device).requires_grad_()
        B = torch.tensor([[-0.3033, -0.4966,  0.0820],
                          [-0.9567,  1.0006, -0.9712],
                          [ 0.0227, -0.6663,  0.2731]], device=device).requires_grad_()

        fname = os.path.join(args.save, 'pypose losses.csv')
        loss_f = open(fname, 'w')
        loss_f.write('im_loss,mse\n')
        loss_f.flush()
        fname = os.path.join(args.save, 'pypose time.csv')
        time_d = open(fname, 'w')
        time_d.write('backward_time,overall_time\n')
        time_d.flush()

        def get_loss(x_init, _A, _B):
            lti = pp.module.LTI(expert['A'], expert['B'], C, D, c1, c2)
            mpc_expert = pp.module.MPC(lti, T, step=1)
            x_true, u_true, cost_true = mpc_expert.forward(expert['Q'], expert['p'], x_init, dt)

            lti_ = pp.module.LTI(_A, _B, C, D, c1, c2)
            mpc_agent = pp.module.MPC(lti_, T, step=1)
            x_pred, u_pred, cost_pred = mpc_agent.forward(expert['Q'], expert['p'], x_init, dt)

            traj_loss = torch.mean((u_true - u_pred)**2) \
                + torch.mean((x_true - x_pred)**2)

            return traj_loss

        opt = optim.RMSprop((A, B), lr=1e-2)

        for i in range(1400):
            t1 = time.time()
            x_init = torch.randn(n_batch, n_state, device=device)
            traj_loss = get_loss(x_init, A, B)
            opt.zero_grad()
            t2 = time.time()
            traj_loss.backward()
            t3 = time.time()
            backward_time = t3 - t2
            opt.step()
            t4 = time.time()
            overall_time = t4 - t1

            model_loss = torch.mean((A - expert['A'])**2) + \
                        torch.mean((B - expert['B'])**2)

            loss_f.write('{},{}\n'.format(traj_loss.item(), model_loss.item()))
            loss_f.flush()
            time_d.write('{},{}\n'.format(backward_time, overall_time))
            time_d.flush()

            plot_interval = 100
            if i % plot_interval == 0:
                print(A, expert['A'])
                print(B, expert['B'])

            print('{:04d}: traj_loss: {:.4f} model_loss: {:.4f}'.format(
                i, traj_loss.item(), model_loss.item()))
            print("backward_time = ", backward_time)
            print("overall_time = ", overall_time)


if __name__=='__main__':
    train = TrainMPC()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train.main(device)
