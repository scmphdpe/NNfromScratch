import numpy as np

def sinf(x, deriv=False):
    if deriv==True:
        return np.cos(x)
    return np.sin(x)

def cosf(x, deriv=False):
    if deriv==True:
        return (-1)*np.sin(x)
    return np.cos(x)

n = 1000 # size of the data set
a = 2 # number of curated activation functions
np.random.seed(59)

# labeled data
x = np.array(np.random.random((n,1)))
y = np.array((np.sin(x),1))

# weight layers
w0 = np.array(np.random.random((a,1)))
w1 = np.array(np.random.random((a,1)))

# input layer
l0 = x

aIn = np.dot(l0, w0.T)

# activation
aMap = np.array((sinf(aIn[:,0]),cosf(aIn[:,1])))

out = np.dot(aMap.T, w1)

loss = (out - y)**2
delta_w1 = 2*loss
w1 = np.dot(delta_w1,w1)
print(w1)
#delta_w0 = np.array((sinf(w1[:,0],deriv=True),cosf(w1[:,0],deriv=True)))
#w0 = np.dot(delta_w0,w0)

print(loss)
