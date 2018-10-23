# following along with Siraj Raval demo from YouTube
# bare bones neural network using only numpy

import numpy as np

def sinf(x, deriv=False):
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
np.random.seed(59)

# synapse (weight) initialization with bias
syn0 = np.random.random((3,1))+1
syn1 = np.random.random((3,1))+1
syn2 = np.random.random((3,1))+1

# training
for j in range(n_data):
    # layers
    l0 = x[j] # input
    l1 = np.array([[sinf(l0)],
                   [cosf(l0)],
                   [x2f(l0)]])

    #l1 = np.dot(l1in, syn0)

    # output layer
    l2 = np.dot(l1, syn0)

    #backpropagation
    l2_error = y[j] - l2
    l2_delta = np.array(l2_error*np.array([sinf(l2, deriv=True),
        cosf(l2, deriv=True),
        x2f(l2, deriv=True)]))
    l1_error = np.dot(l2_delta,syn0.T)
    #l1_delta = np.array(l1_error*np.array([sinf(l1, deriv=True),
    #                                        cosf(l1, deriv=True),
    #                                        x2f(l1, deriv=True)))

    # update synapses
    syn1 += np.dot(l1.T,l2_delta)
    syn0 += np.dot(l0.T,l1_error)

print('output after training')
print(l2)
