"""
    numpy implementation of Multi-Level Perceptron with Tensor.
    Most important concepts are from PyTorch while this one is really premature.
"""

import numpy as np

class Tensor():
    """
        In this project, a Tensor is initialized with a numpy ndarray.
        It has several parameters:
            data: the data of the Tensor, a numpy ndarray
            requires_grad: if this Tensor need to be involved in back propagation, boolean
            grad: the gradient of the Tensor, a numpy ndarray, initialized by 0
            old_grad: used for momentum approach, should be deprecated
        Embedded Functions:
            clear_grad: clear (set to 0) the gradient and old gradient of this Tensor.
    """

    def __init__(self, data, requires_grad = True):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.data.shape) if self.requires_grad else None
        self.old_grad = np.zeros(self.data.shape) if self.requires_grad else None
    
    def clear_grad(self):
        self.grad = np.zeros(self.data.shape) if self.requires_grad else None
        self.old_grad = np.zeros(self.data.shape) if self.requires_grad else None

class FC_layer():
    """
        A standard fully connected layer.
        This layer is initalized by its number of input and output channels.
        Parameters:
            self.w: weight of this layer, a trainable Tensor
            self.b: bias of this layer, a trainable Tensor
            self.input: the input of the last forward graph, preserved to calculate the gradient
        Functions:
            forward: the forward inference of this layer.
                Input: batched input of this layer. Output: batched result of this layer.
            backward: back propagation of this layer.
                Input: propagated error of last layer. Changes: the gradient of weight and bias.
                Output: the propagated error of this layer. 
    """

    def __init__(self, in_channels, out_channels):

        self.w = Tensor(np.random.randn(in_channels,out_channels))
        self.b = Tensor(np.zeros([1, out_channels]))
        self.input = Tensor(np.zeros([1, in_channels]))
    
    def forward(self, the_input):
        self.input.data = the_input
        return np.matmul(self.input.data, self.w.data) + self.b.data
    
    def backward(self, pGrad):
        self.w.grad = np.matmul(self.input.data.transpose(), pGrad)
        self.b.grad = pGrad
        self.input.grad = np.matmul(pGrad, np.transpose(self.w.data))
        return self.input.grad

class ReLU():
    """
        Rectified Linear Unit (ReLU): y = max(x,0)
    """

    def forward(self, the_input):
        self.back = (the_input > 0).astype(float)
        return the_input * self.back
    def backward(self, pGrad):
        return self.back * pGrad

class SoftMax():
    """
        Input: an array of N numbers.
        p = exp(x_i)/(\\sigma_{j=1}^{N} exp(x_j)).
        A mathematical transformation is made to make this function more numerically stable.
    """

    def forward(self, the_input):
        eps = pow(2,-20)
        shift_x = the_input - np.max(the_input)
        the_exp = np.exp(shift_x)
        the_sum = the_exp.sum()
        self.output = the_exp/(the_sum+eps)
        return self.output

    def backward(self, pGrad):
        return self.output * (pGrad - (pGrad*self.output).sum())

class MLP():
    """
        A number of stacked fully connected layers.
        Initalized by:
            1. a config list: [(in_channel, out_channel), ... ,(in_channel, out_channel)] of each layer.
            2. an activation layer.
        Each layer except the last layer is followed by an activation layer.
        The last layer is followed by a SoftMax layer.
    """

    def __init__(self, cfg_list, activation=ReLU):
        self.module_list = []
        for i, item in enumerate(cfg_list):
            self.module_list.append(FC_layer(item[0], item[1]))
            if i != len(cfg_list) - 1:
                self.module_list.append(activation())
            else:
                self.module_list.append(SoftMax())
    
    def forward(self, x):
        for module in self.module_list:
            x = module.forward(x)
        return x

    def backward(self, l):
        for module in reversed(self.module_list):
            l = module.backward(l)
        return l


def MSE_loss(GT, output):
    """ 
        Mean square error loss.
        Input: ground truth and output.
        E = sum((GT - output)^2)
        Output: back propagated error of MSE and the E.
    """

    me = (np.array(output) - np.array(GT))
    return  (me ** 2).sum(), me * 2

def Cross_entropy_loss(GT, output):
    """ 
        Cross entropy loss.
        Input: ground truth and output.
        Output: back propagated error of CE and the CE data.
    """
    eps= pow(2,-10)
    ce = GT * np.log(output+eps)
    return ce.sum() * -1., ce.sum() * GT

class Optimizer():
    """
        w = w - lr * (gradient + momentum * old_grad)
    """

    def __init__(self, model, momentum, lr, schedule = None):
        self.Tensor_list = []
        self.momentum = momentum
        self.lr = lr
        self.schedule = schedule
        self.steps = 0
        for module in model.module_list:
            if isinstance(module, FC_layer):
                self.Tensor_list.append(module.w)
                self.Tensor_list.append(module.b)
        for tensor in self.Tensor_list:
            tensor.old_grad = tensor.grad
    
    def step(self):
        if self.schedule != None:
            self.steps += 1
            if self.steps in self.schedule:
                lr = lr / 10
        for tensor in self.Tensor_list:
            tensor.data = tensor.data - self.lr * ((1- self.momentum) * tensor.grad + self.momentum * tensor.old_grad)

if __name__ == "__main__":
    # Generate input and ground truth
    theInput = np.random.rand(1,3)
    GT = np.array([0.,0.,0.,1.])
    
    # Build model
    mlp = MLP([(3,4),(4,5),(5,4)], ReLU)
    
    # Set training parameters
    lr = 1e-2
    criterion = Cross_entropy_loss
    optimizer = Optimizer(mlp, 0.1, lr)
    iterations = 100000

    # Training monitoring with tqdm (if installed)
    try:
        import tqdm
        loader = tqdm.tqdm(range(iterations))
    except:
        loader = range(iterations)
    
    # collect loss history
    loss_list = []

    # Training
    for i in loader:
        output = mlp.forward(theInput)
        loss, loss_grad = criterion(GT, output)
        _ = mlp.backward(loss_grad)
        loss_list.append(loss)
        if i%(len(loader)/100) == 0:
            lr = lr
        optimizer.step()

    # Visualization of training loss history
    from matplotlib import pyplot as plt
    print(output)
    plt.plot(loss_list)
    plt.show()