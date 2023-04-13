import pypose as pp
import torch as torch

# A temp test file
if __name__=="__main__":
    input_pc = torch.randn([10, 20, 3])
    transT = 0.1 * torch.randn([10, 1, 3])
    print("The ori transT is", transT)
    transR = pp.randn_SO3(10)
    print("The ori transR is", transR)
    transR = transR.matrix()

    output_pc = (input_pc @ transR) + transT

    icpsvd = pp.module.ICP(matched=False)
    result = icpsvd(input_pc, output_pc)
    print("The output is", result)

    # output_transT = result[:,0:3]
    # output_transR = result[:,3:7]
    # output_transR = output_transR.matrix()
    # print("The ICP result is", output_transT.shape)
    # print("The ICP result is", output_transR)
    # # if torch.eq(output_transR, transR) :
    # #     print("pass the test")
