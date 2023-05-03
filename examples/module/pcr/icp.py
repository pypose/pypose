import argparse, os
import pypose as pp
from pointcloud import Pointcloud, pointcloud_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICP Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
    parser.add_argument("--steps", type=int, default=200,
                        help="maximum number of ICP will step, default: 200")
    parser.add_argument("--patience", type=int, default=5,
                        help="number of steps with no loss 'decreasing' is seen, default: 5")
    parser.add_argument("--decreasing", type=float, default=1e-3,
                        help="relative loss decreasing used to count the number of patience steps, default: 1e-3")
    parser.add_argument("--verbose", dest='verbose', action='store_true',
                        help="print a message for each step, default: False")
    parser.set_defaults(verbose=False)
    parser.add_argument("--save", type=str, default='./examples/module/pcr/save/',
                        help="location of png files to save")
    parser.add_argument('--show', dest='show', action='store_true',
                        help="show plot, default: False")
    parser.set_defaults(show=False)
    args = parser.parse_args(); print(args)
    os.makedirs(os.path.join(args.save), exist_ok=True)

    source, target = Pointcloud.load_pointcloud()
    source = source.to(args.device)
    tf = pp.SE3([-0.0500, -0.0200,  0.0000, 0, 0, 0.4794255, 0.8775826]).to(args.device)
    target = tf.Act(target.to(args.device))
    stepper = pp.utils.ReduceToBason(steps=args.steps, patience=args.patience,
                                     decreasing=args.decreasing, verbose=args.verbose)
    icp = pp.module.ICP(stepper=stepper).to(args.device)
    result = icp(source, target)
    reg = result.Act(source)

    pointcloud_plot('ICP', source, target, reg, elev=90, azim=-90,
                    save=args.save, show=args.show)
