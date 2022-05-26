import torch
import pypose as pp

N = 10
# test logic: 
# generate N random sample x0 with ltype
# perform T = x0.matrix()
# x1 = pp.mat2xxx(T)
# compare x0 and x1 are the same LieTensor

# TODO: should explain the result with illegal input in the doc
# z = torch.zeros([3,3])
# z[0,2]=2000
# z.unsqueeze(0)
# print(z)
# print(pp.mat2SO3(z))


def align_quaternion_sign(X0, X1, start_dim):
    assert(X0.shape==X1.shape)
    for i in range(X0.shape[0]):
        if torch.sign(X0[i][start_dim]) != torch.sign(X1[i][start_dim]):
            X1[i][start_dim:start_dim+4] = - X1[i][start_dim:start_dim+4]

def get_different_lines(X0, X1):
    assert(X0.shape==X1.shape)
    for i in range(X0.shape[0]):
        if not torch.allclose(X0[i], X1[i]):
            print(X0[i])
            print(X1[i])

print("Number of Samples: {}".format(N))

rand_generate_funcs = [pp.randn_SO3, pp.randn_SE3, pp.randn_Sim3, pp.randn_RxSO3]
mat2x_funcs = [pp.mat2SO3, pp.mat2SE3, pp.mat2Sim3, pp.mat2RxSO3]
quat_start = [0,3,3,0]

for randn, mat2x, q_start in zip(rand_generate_funcs, mat2x_funcs, quat_start):
    print("\nTest {}".format(mat2x.__name__))
    X0 = randn(N)
    T0 = X0.matrix()
    X1 = mat2x(T0)
    align_quaternion_sign(X0,X1,q_start)
    if not torch.allclose(X0,X1,atol=1e-6):
        get_different_lines(X0, X1)
    else:
        print("All test cases passed")

    # pypose.from_matrix()
    print("Test pp.from_matrix")

    X2 = pp.from_matrix(T0)
    align_quaternion_sign(X0,X2,q_start)
    if not torch.allclose(X0,X1,atol=1e-6):
        get_different_lines(X0, X2)
    else:
        print("All test cases passed")

    # pypose.matrix() (pp.matrix(x) is equivalent to x.matrix())
    T1 = pp.matrix(X0)
    assert torch.allclose(T0, T1)