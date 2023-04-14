import pypose as pp
import torch as torch

# A temp test file
if __name__=="__main__":
    input_pc = torch.randn([10, 20, 3])
    tf = pp.randn_SE3(10)
    print("The true tf is", tf)
    output_pc = tf.unsqueeze(-2).Act(input_pc)
    icpsvd = pp.module.ICP(matched=False)
    result = icpsvd(input_pc, output_pc)
    print("The output is", result)
