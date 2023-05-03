import argparse, os
import pypose as pp
from pointcloud import load_bunny, pointcloud_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICP Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--steps", type=int, default=200,
                        help="maximum number of ICP will step, default: 200")
    parser.add_argument("--patience", type=int, default=5,
                        help="steps with no loss 'decreasing' is seen, default: 5")
    parser.add_argument("--decreasing", type=float, default=1e-3,
                        help="relative loss decreasing used to count the number of \
                            patience steps, default: 1e-3")
    parser.add_argument("--verbose", dest='verbose', action='store_true',
                        help="print a message for each step, default: False")
    parser.add_argument("--dataroot", type=str, default='./examples/module/pcr/data',\
                        help="dataset location")
    parser.add_argument("--save", type=str, default='./examples/module/pcr/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(verbose=False)
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    stepper = pp.utils.ReduceToBason(steps=args.steps, patience=args.patience,
                                     decreasing=args.decreasing, verbose=args.verbose)
    icp = pp.module.ICP(stepper=stepper).to(args.device)

    source, target, gt = load_bunny(root=args.dataroot, device=args.device)
    est = icp(source, target)
    reg = est @ source

    print("Ground Truth:", gt)
    print("ICP Estimate:", est)
    print("Pose Error:", gt.Inv() @ est)
    pointcloud_plot('Bunny-before-ICP', source, target, save=args.save, show=args.show)
    pointcloud_plot('Bunny-after-ICP', reg, target, save=args.save, show=args.show)
