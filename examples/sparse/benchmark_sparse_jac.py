import torch
import pypose as pp
from bal_loader import build_pipeline
from pypose.sparse import sbktensor
import time

TARGET_DATASET = "ladybug"
TARGET_PROBLEM = "problem-49-7776-pre"

def filter_problem(x):
  return x['problem_name'] == TARGET_PROBLEM

def reprojerr_vmap(pose, point, pixel, intrinsic, distortion):
  # reprojerr_vmap is not batched, it operates on a single 3D point and camera
  pose = pp.LieTensor(pose, ltype=pp.SE3_type) # pose will lose its ltype through vmap, temporary fix
  point = pose.unsqueeze(-2) @ point
  point = point.squeeze(-2)

  # perspective division
  point_proj = -point[:2] / point[-1:]

  # convert to pixel coordinates
  f = intrinsic[0, 0]
  k1 = distortion[0]
  k2 = distortion[1]
  n = torch.sum(point_proj**2, dim=-1)
  r = 1.0 + k1 * n + k2 * n**2
  img_repj = f * r * point_proj

  # calculate the reprojection error
  loss = (img_repj - pixel).norm(dim=-1)

  return loss

def construct_sbt(jac_from_vmap, num_cameras, camera_index):
  camera_index = torch.from_numpy(camera_index) # for torch.stack
  n = camera_index.shape[0] # num 2D points
  i = torch.stack([torch.arange(n), camera_index, torch.zeros(n)])
  v = jac_from_vmap[:, None, None, :] # adjust dimension to accomodate for sbt constructor
  return pp.sbktensor(i, v, size=(n, num_cameras, 1), dtype=torch.float32)

def jacrev_custom(func, argnums):
  def wrapper(*args, **kwargs):
    jac_vmap = torch.vmap(pp.func.jacrev(func)) # vmap
    gradients = jac_vmap(*args)
    sbt = construct_sbt(gradients, kwargs['num_cols'], kwargs['index'])
    return sbt
  return wrapper

def benchmark_camera(dataset, n_iter=100):
    start = time.time()
    jac_function_custom = jacrev_custom(reprojerr_vmap, argnums=0)

    while n_iter:
        jac_from_custom = jac_function_custom(dataset['camera_extrinsics'][dataset['camera_index_of_observations']],
                                dataset['points_3d'][dataset['point_index_of_observations'], None],
                                dataset['points_2d'],
                                dataset['camera_intrinsics'][dataset['camera_index_of_observations']],
                                dataset['camera_distortions'][dataset['camera_index_of_observations']],
                                num_cols=len(dataset['camera_extrinsics']),
                                index=dataset['camera_index_of_observations'])
        n_iter -= 1
    end = time.time()
    elapsed = end - start
    return elapsed

def benchmark_points3d(dataset, n_iter=100):
    start = time.time()
    jac_function_custom = jacrev_custom(reprojerr_vmap, argnums=1)

    while n_iter:
        jac_from_custom = jac_function_custom(dataset['camera_extrinsics'][dataset['camera_index_of_observations']],
                                dataset['points_3d'][dataset['point_index_of_observations'], None],
                                dataset['points_2d'],
                                dataset['camera_intrinsics'][dataset['camera_index_of_observations']],
                                dataset['camera_distortions'][dataset['camera_index_of_observations']],
                                num_cols=len(dataset['points_3d']),
                                index=dataset['point_index_of_observations'])
        n_iter -= 1
    end = time.time()
    elapsed = end - start
    return elapsed
  
if __name__ == '__main__':
    dataset_pipeline = build_pipeline(dataset=TARGET_DATASET, cache_dir='bal_data').filter(filter_problem)
    dataset_iterator = iter(dataset_pipeline)
    dataset = next(dataset_iterator)
    elapsed_camera = benchmark_camera(dataset)
    print(f"Elapsed time for camera: {elapsed_camera:.3f} seconds")
    elapsed_points3d = benchmark_points3d(dataset)
    print(f"Elapsed time for points_3d: {elapsed_points3d:.3f} seconds")


