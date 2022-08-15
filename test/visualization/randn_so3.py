import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pypose as pp
from tqdm import tqdm

fig = plt.figure()
ax = Axes3D(fig)

base_point = np.array([0, 0, 1])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
n = 5000
xs = np.zeros(n)
ys = np.zeros(n)
zs = np.zeros(n)

for i in tqdm(range(n)):
    phi = pp.randn_so3(1).numpy()
    theta = np.sqrt(np.sum(np.square(phi)))
    n = phi/theta
    Rotate_Matrix = np.cos(theta)*np.identity(3) + \
                    (1-np.cos(theta))*(n.T @ n)+ \
                    np.sin(theta)*np.matrix([
        [      0, -n[0][2],  n[0][1]],
        [ n[0][2],       0, -n[0][0]],
        [-n[0][1],  n[0][0],       0]
    ])
    sample_points = Rotate_Matrix @ base_point.T
    xs[i] = sample_points[0, 0]
    ys[i] = sample_points[0, 1]
    zs[i] = sample_points[0, 2]

ax.scatter(xs, ys, zs, c='blue', marker='.', s=10)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
plt.savefig("3D.svg", bbox_inches='tight')