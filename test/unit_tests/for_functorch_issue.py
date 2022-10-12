import torch
from torch.utils._pytree import tree_map
from functorch import vmap, jacrev

# extension_functions = [ 'my_func' ]

# class MyTensor(torch.Tensor):
#     def __init__(self, *args, **kwargs):
#         pass

#     @staticmethod
#     def __new__(cls, *args, **kwargs):
#         return super().__new__(cls, *args, **kwargs)

#     @classmethod
#     def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
#         def unwrap(x):
#             print(x.__class__.__bases__[0])
#             return x.__class__.__bases__[0] if isinstance(x, MyTensor) else x

#         rs = func( *tree_map(unwrap, args), **tree_map(unwrap, kwargs) )

#         return MyTensor(rs)

#     def my_func(self):
#         return self

# if __name__ == '__main__':
#     x = MyTensor((2,))
#     y = torch.Tensor((42,))

#     # import ipdb; ipdb.set_trace()
#     # x.my_func()
#     a = x + y

#     # f = lambda a, b: a.my_func() * b
#     # print(f(x, y))

#     # # This triggers the error.
#     # j = jacrev(f)(x, y)

#     # # The followings are OK.
#     # x = torch.Tensor((2,))
#     # f = lambda a, b: a * b
#     # j = jacrev(f)(x, y)

#     # print(j)

#     # x = MyTensor( ((1, ), (2, )) )
#     # y = torch.Tensor( ((42,), (42,)) )

#     # f = lambda a, b: a.my_func() * b
#     # print(f(x,y))

#     # v = vmap(f)(x, y)
#     # print(v)
