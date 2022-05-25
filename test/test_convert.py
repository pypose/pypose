import torch
import pypose as pp

N = 10000
# test logic: 
# generate N random sample x0 with ltype
# perform T = x0.matrix()
# x1 = pp.mat2xxx(T)
# compare x0 and x1 are the same LieTensor


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
# pypose.mat2SO3
print('Testing mat2SO3')

x0 = pp.randn_SO3(N)
T = x0.matrix()
x1 = pp.mat2SO3(T)
align_quaternion_sign(x0,x1,0)
if not torch.allclose(x0,x1,atol=1e-6):
    get_different_lines(x0, x1)
else:
    print("All test cases passed")


# pypose.mat2SE3
print('Testing mat2SE3')

x0 = pp.randn_SE3(N)
T = x0.matrix()
x1 = pp.mat2SE3(T)
align_quaternion_sign(x0,x1,3)
if not torch.allclose(x0,x1,atol=1e-6):
    get_different_lines(x0, x1)
else:
    print("All test cases passed")


# pypose.mat2Sim3()
print('Testing mat2Sim3')

x0 = pp.randn_Sim3(N)
T = x0.matrix()
x1 = pp.mat2Sim3(T)
align_quaternion_sign(x0,x1,3)
if not torch.allclose(x0,x1,atol=1e-6):
    get_different_lines(x0, x1)
else:
    print("All test cases passed")



# pypose.mat2RxSO3()
print('Testing mat2RxSO3')

x0 = pp.randn_RxSO3(N)
T = x0.matrix()
x1 = pp.mat2RxSO3(T)
align_quaternion_sign(x0,x1,0)
if not torch.allclose(x0,x1,atol=1e-6):
    get_different_lines(x0, x1)
else:
    print("All test cases passed")


# pypose.from_matrix()

# pypose.matrix() (pp.matrix(x) is equivalent to x.matrix())