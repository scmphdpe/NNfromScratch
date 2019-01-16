# following along with Siraj Raval demo from YouTube
# bare bones neural network using only numpy

import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input data
x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# seed
np.random.seed(1)

# synapse initialization
syn0 = 2*np.random.random((3,4))-1
syn1 = 2*np.random.random((4,1))-1

# training
for j in range(60000):
    # layers
    l0 = x
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    #backpropagation
    l2_error = y - l2

    if (j % 10000) == 0:
        print('Error:' + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error*nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error*nonlin(l1, deriv=True)

    # update synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print('output after training')
print(l2)
