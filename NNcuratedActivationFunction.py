# following along with Siraj Raval demo from YouTube
# bare bones neural network using only numpy

import numpy as np

# activation - sigmoid function
def sigmoid(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

def sinef(x, deriv=False):
    if (deriv==True):
        return np.cos(x)
    return np.sin(x)

def cosf(x, deriv=False):
    if deriv==True:
        return (-1.)*np.sin(x)
    return np.cos(x)

def x2f(x, deriv=True):
    if deriv==True:
        return 2*x
    return x**2

# input data
n_data = 1000
x = np.array(np.random.rand(n_data))

y = np.array(x**2)

# seed
np.random.seed(1)

# synapse (weight) initialization with no bias
syn0 = np.random.random((4,1))
syn1 = np.random.random((4,1))
syn2 = np.random.random((4,1))
syn3 = np.random.random((4,1))

# training
for j in range(n_data):
    # layers
    l0 = x[j] # input
    l1in = np.array([sigmoid(l0),
                   sinef(l0),
                   cosf(l0),
                   x2f(l0)])
    l1 = np.dot(l1in, syn0)
    # output layer
    l2 = np.dot(l1, syn1)

    #backpropagation
    l2_error = y[j] - l2
    l2_delta = np.array(l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1, deriv=True)

    # update synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print('output after training')
print(l2)
