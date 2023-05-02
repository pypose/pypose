import argparse, os
import pypose as pp
from pointcloud import Pointcloud, pointcloud_plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICP Example')
    parser.add_argument("--device", type=str, default='cpu', help="cuda or cpu")
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
    icp = pp.module.ICP().to(args.device)
    result = icp(source, target)
    reg = result.Act(source)

    pointcloud_plot('ICP', source, target, reg, elev=90, azim=-90,
                    save=args.save, show=args.show)
