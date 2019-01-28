# following along with Siraj Raval demo from YouTube
# bare bones neural network using only numpy

import numpy as np

# sigmoid function
def caf(x, deriv=False):
    if(deriv==True):
        return 2*x
    return x**2

def trueFunc(x):
    return caf(x)

def taf(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# seed
np.random.seed(1)

# synapse initialization
syn = 2*np.random.random((1,2))-1

# training
for j in range(1000):

    if (j % 500) == 0:
        print(syn)
        print(f'synapse shape = {syn.shape}')
    # layers
    l0 = 3 #np.array([np.random.ranf()*100])
    y_true = trueFunc(l0)

    if (j % 500) == 0:
        print(f'l0 = {l0}')
        #print(f'input shape = {l0.shape}')

    i0 = np.dot(l0,syn)

    if (j % 500) == 0:
        print(i0)
        print(f'i0 shape = {i0.shape}')

    l1 = np.array([caf(i0[0]),taf(i0[1])])

    if (j % 500) == 0:
        print(l1)
        print(f'l1 shape is {l1.shape}')

    y_hat = np.sum(l1)

    # backpropagation
    residual = trueFunc(l0) - y_hat

    if (j % 500) == 0:
        print(f'in = {l0}, y_hat = {y_hat}, y_true = {y_true}, residual={residual}')
        #print(f'residual shape is {residual.shape}')

    l1_back = np.array([caf(l1[0],deriv=True), taf(l1[1],deriv=True)]).T
    if (j % 500) == 0:
        print(l1_back)
        print(f'l1_back shape is {l1_back.shape}')

    delta = residual*l1_back
    if (j % 500) == 0:
        print(delta)
        print(f'delta shape is {delta.shape}')
    syn += l1*delta

print(f'Final weights after training are {syn}')
