from pypose.module.cost import Cost
import torch as torch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class nnCost(Cost):
    def __init__(self, hiddenSize):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, hiddenSize[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenSize[0], hiddenSize[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddenSize[1], 1))
    
    def cost(self, state, input):
        print(self.net(state))
        exit()
        return self.net(state)**2 + self.net(input)**2
    

def createTimePlot(x, y, figname="Un-named plot", title=None, xlabel=None, ylabel=None):
    f = plt.figure(figname)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return f

if __name__ == "__main__":
    #  state and input
    state = torch.randn(2, 1, 3)

    input = torch.tensor([1.])

    # Create solver object
    nnSolver = nnCost([5, 10])
    # Calculate cost
    cost = nnSolver.forward(state, input)
    print(cost)
