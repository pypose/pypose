from pypose.lietensor.convert import mat2SE3, mat2SO3, mat2Sim3, mat2RxSO3
import torch
import pypose as pp


# sample input

# rotation 90 degree around the z axis
input = torch.tensor([[0., -1., 0.],
                      [1., 0., 0.],
                      [0., 0., 1.]], dtype = torch.float64,  requires_grad=True)

x = pp.mat2SO3(input)
output = pp.gradcheck(mat2SO3, input)
print(torch.allclose(output[0][0], output[1][0]))
print(x)


input = torch.tensor([[0., -1., 0., 0.1],
                      [1., 0., 0., 0.2],
                      [0., 0., 1., 0.3],
                      [0., 0., 0., 1.]], dtype = torch.float64,  requires_grad=True)
output = pp.gradcheck(mat2SE3, input)
print(torch.allclose(output[0][0], output[1][0]))
x = pp.mat2SE3(input)
print(x)

input = torch.tensor([[0., -0.5, 0., 0.1],
                      [0.5, 0., 0., 0.2],
                      [0., 0., 0.5, 0.3],
                      [0., 0., 0., 1.]], dtype = torch.float64,  requires_grad=True)
output = pp.gradcheck(mat2Sim3, input)
print(torch.allclose(output[0][0], output[1][0]))
x = pp.mat2Sim3(input)
print(x)

input = torch.tensor([[0., -0.5, 0.],
                      [0.5, 0., 0.],
                      [0., 0., 0.5]], dtype = torch.float64,  requires_grad=True)
output = pp.gradcheck(mat2RxSO3, input)
print(torch.allclose(output[0][0], output[1][0]))
x = pp.mat2RxSO3(input)
print(x)


print("Test illegal input:")
z = torch.zeros([3, 3], dtype=torch.float64)
z[0, 2] = 10
print(z)
print(pp.mat2SO3(z, check=True))

x = torch.randn([3, 3], dtype=torch.float64)
print(x)
print(pp.mat2SO3(x, check=True))

print("Test input with more or less information")
a = pp.randn_SO3(dtype=torch.float64)
t = a.matrix()
b = pp.mat2SE3(t)
# print(a,b)
b = pp.mat2Sim3(t)
# print(a, b)

a = pp.randn_SE3(dtype=torch.float64)
t = a.matrix()
b = pp.mat2SO3(t)
# print(a,b)
b = pp.mat2RxSO3(t)
# print(a,b)


N = 100
shape = torch.Size([10, 10])
# test logic:
# generate N random sample x0 with ltype
# perform T = x0.matrix()
# x1 = pp.mat2xxx(T)
# compare x0 and x1 are the same LieTensor


def align_quaternion_sign(X0, X1, start_dim):
    assert(X0.shape == X1.shape)
    for i in range(X0.shape[0]):
        if torch.sign(X0[i][start_dim]) != torch.sign(X1[i][start_dim]):
            X1[i][start_dim:start_dim+4] = - X1[i][start_dim:start_dim+4]


def get_different_lines(X0, X1):
    assert(X0.shape == X1.shape)
    for i in range(X0.shape[0]):
        if not torch.allclose(X0[i], X1[i]):
            print(X0[i])
            print(X1[i])


print("Number of Samples: {}".format(N))

rand_generate_funcs = [pp.randn_SO3,
                       pp.randn_SE3, pp.randn_Sim3, pp.randn_RxSO3]
mat2x_funcs = [pp.mat2SO3, pp.mat2SE3, pp.mat2Sim3, pp.mat2RxSO3]
quat_start = [0, 3, 3, 0]

for randn, mat2x, q_start in zip(rand_generate_funcs, mat2x_funcs, quat_start):
    print("\nTest {}".format(mat2x.__name__))
    x0 = randn(dtype=torch.float64)
    t = x0.matrix()
    x1 = mat2x(t)
    if torch.sign(x0[q_start]) != torch.sign(x1[q_start]):
        x1[q_start:q_start+4] = - x1[q_start:q_start+4]
    # print(x0, x1)
    assert torch.allclose(x0, x1, atol=1e-6)
    print("Test 0 dimension passed")

    X0_raw = randn(N, dtype=torch.float64, requires_grad=True)
    X0 = X0_raw.view(shape + x0.shape)

    T0 = X0.matrix()
    # print(T0.shape)
    X1 = mat2x(T0)
    X1_raw = X1.view(X0_raw.shape)
    align_quaternion_sign(X0_raw, X1_raw, q_start)
    if not torch.allclose(X0_raw, X1_raw, atol=1e-6):
        get_different_lines(X0_raw, X1_raw)
    else:
        print("All test cases passed")

    # pypose.from_matrix()
    print("Test pp.from_matrix")

    X2 = pp.from_matrix(T0, X0.ltype)
    X2_raw = X2.view(X0_raw.shape)
    align_quaternion_sign(X0_raw, X2_raw, q_start)
    if not torch.allclose(X0_raw, X2_raw, atol=1e-6):
        get_different_lines(X0_raw, X2_raw)
    else:
        print("All test cases passed")

    # pypose.matrix() (pp.matrix(x) is equivalent to x.matrix())
    T1 = pp.matrix(X0)
    assert torch.allclose(T0, T1)

    # gradcheck
    output = pp.gradcheck(mat2x, T0)
    if torch.allclose(output[0][0], output[1][0], torch.finfo(T0.dtype).resolution):
        print("Grad check passed")
    else:
        print("Grad check failed")
