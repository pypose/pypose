import torch
import torch.optim as optim
import mpc
import pypose as pp
import time
import os
import shutil
import pickle as pkl
import argparse
import setproctitle

# import setGPU


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_state', type=int, default=5)
    parser.add_argument('--n_ctrl', type=int, default=5)
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = '.'.join(["{}={}".format(x, getattr(args, x))
                  for x in ['n_state', 'n_ctrl', 'T']])
    setproctitle.setproctitle('bamos.lqr.'+t+'.{}'.format(args.seed))
    if args.save is None:
        args.save = os.path.join(args.work, t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = 'cuda' if args.cuda else 'cpu'

    n_state, n_ctrl = args.n_state, args.n_ctrl
    n_sc = n_state+n_ctrl

    expert_seed = 26

    assert expert_seed != args.seed
    torch.manual_seed(expert_seed)

    Q = torch.eye(n_sc)
    p = torch.randn(n_sc)
    C = torch.eye(n_state)
    D = torch.zeros(n_state, n_ctrl)
    c1 = torch.zeros(n_state)
    c2 = torch.zeros(n_state)

    print(Q)

    alpha = 0.2

    expert = dict(
        Q = torch.eye(n_sc).to(device),
        p = torch.randn(n_sc).to(device),
        A = (torch.eye(n_state) + alpha * torch.randn(n_state, n_state)).to(device),
        B = torch.randn(n_state, n_ctrl).to(device),
    )
    fname = os.path.join(args.save, 'expert.pkl')
    with open(fname, 'wb') as fi:
        pkl.dump(expert, fi)

    torch.manual_seed(args.seed)
    A = (torch.eye(n_state) + alpha * torch.randn(n_state, n_state))\
        .to(device).requires_grad_()
    B = torch.randn(n_state, n_ctrl).to(device).requires_grad_()

    u_init = None

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
        _lqr2 = mpc.MPC(
            n_state, n_ctrl, args.T, lti,
            u_init=u_init,
            lqr_iter=1,
            n_batch = n_batch)
        x_true, u_true, objs_true = _lqr2.forward(x_init, expert['Q'], expert['p'])

        lti_ = pp.module.LTI(_A, _B, C, D, c1, c2)
        _lqr2 = mpc.MPC(
            n_state, n_ctrl, args.T, lti_,
            u_init=u_init,
            lqr_iter=1,
            n_batch = n_batch)
        x_pred, u_pred, objs_pred = _lqr2.forward(x_init, expert['Q'], expert['p'])

        traj_loss = torch.mean((u_true - u_pred)**2) + \
                    torch.mean((x_true - x_pred)**2)
        return traj_loss

    opt = optim.RMSprop((A, B), lr=1e-2)

    n_batch = 4
    for i in range(500):

        t1 = time.time()
        x_init = torch.randn(n_batch,n_state).to(device)
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
            os.system('./plot.py "{}" &'.format(args.save))
            print(A, expert['A'])
            print(B, expert['B'])
        print('{:04d}: traj_loss: {:.4f} model_loss: {:.4f}'.format(
            i, traj_loss.item(), model_loss.item()))

        print("backward_time = ", backward_time)
        print("overall_time = ", overall_time)


if __name__=='__main__':
    main()
