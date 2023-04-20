import csv
import torch
import pypose as pp

class TestICP:

    def load_csv_point_cloud():
        # Read the CSV file
        csv_file = './test/module/icp-test-data.csv'

        pc1 = []
        pc2 = []

        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                pc1.append([float(row[0]), float(row[1]), 1])
                if row[2] != '':
                    pc2.append([float(row[2]), float(row[3]), 1])

            pc1_tensor = torch.tensor(pc1)
            pc2_tensor = torch.tensor(pc2)

        return pc1_tensor, pc2_tensor

    def l_shape_wall(b, side1, side2, resolution=0.01, noise_std=1e-4):
        # Generate a Laser Scanning for an L-shape wall
        n_points_side1 = int(side1 / resolution)
        n_points_side2 = int(side2 / resolution)

        # Generate the x and y coordinates for both sides of the wall
        x_side1 = torch.linspace(0, side1 - resolution, n_points_side1)
        y_side1 = torch.full((n_points_side1,), side2)

        x_side2 = torch.full((n_points_side2,), side1)
        y_side2 = torch.linspace(side2, resolution, n_points_side2)
        y_side2 = y_side2.flip(-1)  # Reverse the tensor

        # Concatenate the x and y coordinates for both sides
        x = torch.cat((x_side1, x_side2))
        y = torch.cat((y_side1, y_side2))

        # Set the z coordinates to a constant value of 1
        z = torch.ones_like(x)

        # Stack the coordinates into a single tensor
        pc1 = torch.stack([x, y, z], dim=1).unsqueeze(0).repeat(b, 1, 1)


        # Add random noise to pc1 to generate pc2
        noise = torch.normal(0, noise_std, size=pc1.shape)
        pc2 = pc1 + noise

        tf = pp.randn_SE3(b)
        pc2 = tf.unsqueeze(-2).Act(pc2)

        return pc1, pc2, tf

    def laser_scan():
        pc1, pc2  = TestICP.load_csv_point_cloud()
        tf = pp.randn_SE3(1)
        pc2 = tf.unsqueeze(-2).Act(pc2)
        return pc1, pc2, tf

    def random_point_cloud(b, num_points):
        pc1 = torch.rand(b, num_points, 3)
        tf = pp.randn_SE3(b)
        pc2 = tf.unsqueeze(-2).Act(pc1)
        return pc1, pc2, tf



if __name__ == "__main__":
    b = 2
    side1 = 0.1  # Length of the first side of the L-shaped wall
    side2 = 0.5   # Length of the second side of the L-shaped wall
    resolution = 0.01  # Scanning resolution
    noise_std = 0  # Standard deviation of the noise
    num_points = 20
    # pc1, pc2, tf  = TestICP.l_shape_wall(b, side1, side2, resolution, noise_std)
    # pc1, pc2, tf  = TestICP.random_point_cloud(b, num_points)
    pc1, pc2, tf = TestICP.laser_scan()
    icpsvd = pp.module.ICP()
    result = icpsvd(pc1, pc2)
    print("The true tf is", tf)
    print("The output is", result)


    # import torch, pypose as pp
    # b = torch.randint(low=1, high=10, size=())
    # num_points1 = torch.randint(low=2, high=100, size=())
    # num_points2 = torch.randint(low=2, high=100, size=())
    # pc1 = torch.rand(b, num_points1, 3)
    # pc2 = torch.rand(b, num_points2, 3)
    # print(pc1.shape)
    # print(pc2.shape)
    # dist, idx = pp.module.ICP._k_nearest_neighbor(pc1, pc2, k = 5, sort = True)
    # print(dist.shape)
    # print(idx.shape)
