import os
import time
import torch
import shutil
import argparse
import setproctitle
import pypose as pp
import pickle as pkl
import torch.optim as optim
from invpend import InvPend
from pypose.module.ipddp import ddpOptimizer

def main():
    torch.set_default_dtype(torch.float64)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_state', type=int, default=2)
    parser.add_argument('--n_input', type=int, default=1)
    parser.add_argument('--T', type=int, default=4)
    parser.add_argument('--save', type=str)
    parser.add_argument('--work', type=str, default='work')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    t = '.'.join(["{}={}".format(x, getattr(args, x))
                  for x in ['n_state', 'n_input', 'T']])
    setproctitle.setproctitle('bamos.lqr.'+t+'.{}'.format(args.seed))
    if args.save is None:
        args.save = os.path.join(args.work, 'invpend', t, str(args.seed))

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    device = 'cuda' if args.cuda else 'cpu'

    n_state, n_input = args.n_state, args.n_input
    n_sc = n_state+n_input
    N = args.T
    expert_seed = 26

    assert expert_seed != args.seed
    torch.manual_seed(expert_seed)

    dt = 0.05   # Delta t
    Q = dt*torch.eye(n_state, n_state)
    R = dt*torch.eye(n_input, n_input)
    S = torch.zeros(n_state, n_input)
    c = torch.zeros(1, 1)
    stage_cost = pp.module.QuadCost(Q, R, S, c)
    terminal_cost = pp.module.QuadCost(10./dt*Q, R, S, c)
    
    # Create constraint object
    gx = torch.zeros( 2*n_input, n_state)
    gu = torch.vstack( (torch.eye(n_input, n_input), - torch.eye(n_input, n_input)) )
    g = torch.hstack( (-0.25 * torch.ones(1, n_input), -0.25 * torch.ones(1, n_input)) )
    lincon = pp.module.LinCon(gx, gu, g)


    expert = dict(
        Q = torch.eye(n_sc).to(device),
        p = torch.randn(n_sc).to(device),
        param = torch.Tensor([10.0]).to(device)
    )
    fname = os.path.join(args.save, 'expert.pkl')
    with open(fname, 'wb') as fi:
        pkl.dump(expert, fi)

    torch.manual_seed(args.seed)
    param = torch.Tensor([15.0]).to(device).requires_grad_()

    state = torch.tensor([[-1., 0.]])
    state_all =      torch.zeros(N+1, 1, n_state)
    input_all = 0.02*torch.ones(N,    1, n_input)
    init_traj = {'state': state_all, 
                 'input': input_all}
    state_all[0] = state

    fname = os.path.join(args.save, 'pypose losses.csv')
    loss_f = open(fname, 'w')
    loss_f.write('im_loss,mse\n')
    loss_f.flush()

    fname = os.path.join(args.save, 'pypose time.csv')
    time_d = open(fname, 'w')
    time_d.write('backward_time,overall_time\n')
    time_d.flush()

    def get_loss(_param):
        sys_ = InvPend(dt, length=_param)
        _ipddp = ddpOptimizer(sys_, stage_cost, terminal_cost, lincon, 
                                n_state, n_input, gx.shape[0], 
                                N, init_traj) 
        _fp, _, _ = _ipddp.forward()
        x_pred, u_pred = _fp.x, _fp.u
        print('x_pred solved')

        sys = InvPend(dt, length=expert['param'])
        ipddp = ddpOptimizer(sys, stage_cost, terminal_cost, lincon, 
                                n_state, n_input, gx.shape[0], 
                                N, init_traj) 
        fp, _, _ = ipddp.optimizer()
        x_true, u_true = fp.x, fp.u
        print('x_true solved')

        traj_loss = 1e4*(torch.mean((u_true - u_pred)**2) + \
                    torch.mean((x_true - x_pred)**2))
        return traj_loss

    # opt = optim.RMSprop([param], lr=1e-2)
    opt = optim.SGD([param], lr=2e-1)

    n_batch = 1
    for i in range(50):

        t1 = time.time()
        traj_loss = get_loss(param)
        opt.zero_grad()
        t2 = time.time()
        with torch.autograd.set_detect_anomaly(True):
            traj_loss.backward()
            print('grad', param.grad)
        t3 = time.time()
        backward_time = t3 - t2
        opt.step()
        t4 = time.time()
        overall_time = t4 - t1

        model_loss = torch.mean((param - expert['param'])**2)

        loss_f.write('{},{}\n'.format(traj_loss.item(), model_loss.item()))
        loss_f.flush()

        time_d.write('{},{}\n'.format(backward_time, overall_time))
        time_d.flush()

        plot_interval = 1
        if i % plot_interval == 0:
            print(param, expert['param'])
        print('{:04d}: traj_loss: {:.4f} model_loss: {:.4f}'.format(
            i, traj_loss.item(), model_loss.item()))
    
    print( args.save)
    os.system('python .\plot.py "{}" &'.format(args.save))
if __name__=='__main__':
    main()
