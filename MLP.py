# Deprecated
import numpy as np

class FC_layer():
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = np.random.randn(self.in_channels,self.out_channels)
        self.b = np.zeros([1,self.out_channels])
        self.grad_w = np.zeros([self.in_channels,self.out_channels])
        self.grad_b = np.zeros([1,self.out_channels])
        self.input = np.zeros([1, self.in_channels])
    
    def forward(self, the_input):
        self.input = the_input
        return np.matmul(the_input, self.w) + self.b
    
    def backward(self, pGrad):
        # self.grad_w = self.grad_w / 10. + np.matmul(self.input.transpose(), pGrad)
        # self.grad_b = self.grad_b / 10. + pGrad
        self.grad_w = np.matmul(self.input.transpose(), pGrad)
        self.grad_b = pGrad
        return np.matmul(pGrad, np.transpose(self.w))
    
    def clear_grad(self):
        self.grad_w = np.zeros([self.in_channels,self.out_channels])
        self.grad_b = np.zeros([1,self.out_channels])
    
    def step(self, lr):
        self.w = self.w - lr * self.grad_w
        self.b = self.b - lr * self.grad_b

class ReLU():
    def forward(self, the_input):
        self.back = (the_input > 0).astype(float)
        self.output = the_input * self.back
        return self.output
    def backward(self, pGrad):
        return self.back * pGrad
    
    def step(self, lr):
        pass

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
    
    def step(self, lr):
        for module in self.module_list:
            module.step(lr)

class SoftMax():
    def __init__(self):
        self.eps = 1e-10

    def forward(self, the_input):
        shift_x = the_input - np.max(the_input)
        the_exp = np.exp(shift_x)
        the_sum = the_exp.sum()
        self.output = the_exp/(the_sum+self.eps)
        return self.output

    def backward(self, pGrad):

        return self.output * (pGrad - (pGrad*self.output).sum())
        #return ((self.the_sum-self.the_exp-1)*pGrad - (self.the_exp*pGrad).sum())/((self.the_sum ** 2)+self.eps)
    
    def step(self, lr):
        pass    


def mse_loss(GT, output):
    mse = (np.array(output) - np.array(GT))
    return  (mse ** 2).sum() ,mse * 2 #* np.array(output)

def cel(GT, output):
    eps= 1e-10
    ce = GT * np.log(output+eps) * -1.
    return ce.sum(), (1/(ce+eps)) 

theInput = np.random.rand(1,3)
fc1 = FC_layer(3,4)
act = ReLU()
fc2 = FC_layer(4,4)
lr = 1e-1
GT = [0.,0.,0.,1.]

mlp = MLP([(3,4)])
#print(mlp.module_list[0].forward(theInput))
loss_list = []
theRange = 10000
for i in range(theRange):
    output = mlp.forward(theInput)
    loss, loss_grad = mse_loss(GT, output)
    _ = mlp.backward(loss_grad)
    loss_list.append(loss)
    if i%(theRange/100) == 0:
        lr = lr
        # print(loss, output)
    mlp.step(lr)

from matplotlib import pyplot as plt
print(output)
plt.plot(loss_list)
plt.show()