import numpy as np

class Tensor():
    def __init__(self, data, requires_grad = True):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros(self.data.shape) if self.requires_grad else None
        self.old_grad = np.zeros(self.data.shape) if self.requires_grad else None
    
    def clear_grad(self):
        self.grad = np.zeros(self.data.shape) if self.requires_grad else None
        self.old_grad = np.zeros(self.data.shape) if self.requires_grad else None

class FC_layer():
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
    def forward(self, the_input):
        self.back = (the_input > 0).astype(float)
        return the_input * self.back
    def backward(self, pGrad):
        return self.back * pGrad

class MLP():
    def __init__(self, cfg_list):
        self.module_list = []
        for i, item in enumerate(cfg_list):
            self.module_list.append(FC_layer(item[0], item[1]))
            if i != len(cfg_list) - 1:
                self.module_list.append(ReLU())
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

class SoftMax():
    def forward(self, the_input):
        eps = pow(2,-20)
        shift_x = the_input - np.max(the_input)
        the_exp = np.exp(shift_x)
        the_sum = the_exp.sum()
        self.output = the_exp/(the_sum+eps)
        return self.output

    def backward(self, pGrad):
        return self.output * (pGrad - (pGrad*self.output).sum())

def mse_loss(GT, output):
    mse = (np.array(output) - np.array(GT))
    return  (mse ** 2).sum(), mse * 2

def cel(GT, output):
    eps= pow(2,-10)
    ce = GT * np.log(output+eps)
    return ce.sum() * -1., ce.sum() * GT

class Optimizer():
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
    
    theInput = np.random.rand(1,3)
    fc1 = FC_layer(3,4)
    act = ReLU()
    fc2 = FC_layer(4,4)
    lr = 1e-2
    GT = np.array([0.,0.,0.,1.])

    mlp = MLP([(3,4),(4,5),(5,4)])
    #exit(0)
    #print(mlp.module_list[0].forward(theInput))
    loss_list = []
    theRange = 100000
    # o = Optimizer(mlp, 0.1, lr, range(theRange, int(theRange/10)))
    o = Optimizer(mlp, 0.1, lr)
    import tqdm
    loader = tqdm.tqdm(range(theRange))
    for i in loader:
        output = mlp.forward(theInput)
        loss, loss_grad = cel(GT, output)
        _ = mlp.backward(loss_grad)
        loss_list.append(loss)
        if i%(theRange/100) == 0:
            lr = lr
            # print(loss, output)
        o.step()

    from matplotlib import pyplot as plt
    print(output)
    plt.plot(loss_list)
    plt.show()