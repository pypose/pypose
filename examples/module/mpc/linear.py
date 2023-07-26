import os
import time
import torch
import argparse
import pypose as pp
import torch.optim as optim
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='MPC Linear Learning Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--save", type=str, default='./examples/module/mpc/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    n_batch, n_state, n_ctrl, T = 5, 3, 3, 5
    n_sc = n_state + n_ctrl

    C = torch.eye(n_state, device=args.device)
    D = torch.zeros(n_state, n_ctrl, device=args.device)
    c1 = torch.zeros(n_state, device=args.device)
    c2 = torch.zeros(n_state, device=args.device)
    dt = 1

    # expert
    exp = dict(
        Q = torch.eye(n_sc, device=args.device).tile(n_batch, T, 1, 1),
        p = torch.tensor([ 0.6336, -0.2203, -0.1395, -0.7664,  0.8874,  0.8153], \
                                    device=args.device).tile(n_batch, T, 1),
        A = torch.tensor([[ 1.1267, -0.0441, -0.0279],
                            [-0.1533,  1.1775,  0.1631],
                            [ 0.1618,  0.1238,  0.9489]], device=args.device),
        B = torch.tensor([[ 0.4567,  0.7805,  0.0319],
                            [-0.5938, -0.5724,  0.0422],
                            [-0.1804, -0.2535,  1.7218]], device=args.device))
    # Based on n_batch, n_state and n_ctrl, different Q, p, A, B can be given.
    # Note that the given A and B should make the system controllable.

    torch.manual_seed(0)
    A = torch.tensor([[ 1.2082, -0.1587, -0.3358],
                        [ 0.2137,  0.8831, -0.1797],
                        [ 0.1807,  0.2676,  0.7561]], device=args.device).requires_grad_()
    B = torch.tensor([[-0.3033, -0.4966,  0.0820],
                        [-0.9567,  1.0006, -0.9712],
                        [ 0.0227, -0.6663,  0.2731]], device=args.device).requires_grad_()

    def get_loss(x_init, _A, _B):
        lti = pp.module.LTI(exp['A'], exp['B'], C, D, c1, c2)
        stepper_exp = pp.utils.ReduceToBason(steps=1, verbose=False)
        mpc_exp = pp.module.MPC(lti, exp['Q'], exp['p'], T, stepper=stepper_exp) # expert
        x_true, u_true, cost_true = mpc_exp.forward(dt, x_init)

        lti_ = pp.module.LTI(_A, _B, C, D, c1, c2)
        stepper_agt = pp.utils.ReduceToBason(steps=1, verbose=False)
        mpc_agt = pp.module.MPC(lti_, exp['Q'], exp['p'], T, stepper=stepper_agt) # agent
        x_pred, u_pred, cost_pred = mpc_agt.forward(dt, x_init)

        traj_loss = torch.mean((u_true - u_pred)**2) \
            + torch.mean((x_true - x_pred)**2)

        return traj_loss

    opt = optim.RMSprop((A, B), lr=1e-2)

    steps = 1400
    traj_losses = []
    model_losses = []

    for i in range(steps):
        t1 = time.time()
        x_init = torch.randn(n_batch, n_state, device=args.device)
        traj_loss = get_loss(x_init, A, B)
        opt.zero_grad()
        t2 = time.time()
        traj_loss.backward()
        t3 = time.time()
        backward_time = t3 - t2
        opt.step()
        t4 = time.time()
        overall_time = t4 - t1

        model_loss = ((A - exp['A'])**2).mean() + ((B - exp['B'])**2).mean()

        traj_losses.append(traj_loss.item())
        model_losses.append(model_loss.item())

        plot_interval = 100
        if i % plot_interval == 0:
            print(A, exp['A'])
            print(B, exp['B'])

        print('{:04d}: traj_loss: {:.4f} model_loss: {:.4f}'.format(
            i, traj_loss.item(), model_loss.item()))
        print("backward_time = ", backward_time)
        print("overall_time = ", overall_time)

    plt.subplot(2, 1, 1)
    plt.plot(range(steps), traj_losses)
    plt.ylabel('Trajectory Loss')

    plt.subplot(2, 1, 2)
    plt.plot(range(steps), model_losses)
    plt.xlabel('Iteration')
    plt.ylabel('Model Loss')

    figure = os.path.join(args.save + 'lineaer.png')
    plt.savefig(figure)
    print("Saved to", figure)

    if args.show:
        plt.show()
