# perceptAND

import numpy as np
from numpy import random

# Inputs for perceptAND
x1 = [0, 0, 1, 1] 
x2 = [0, 1, 0, 1]
Yd = [0, 0, 0, 1]

# Initialization - set theta to random number [-0.5, 0.5]
theta = round(random.uniform(-0.5, 0.5), 1)
alpha = 0.1
n_epoch = 5

def perceptAND(x1, x2, Yd, theta, alpha, n_epoch):
    # Initialization - set initial weights to random numbers [-0.5, 0.5]
    w1 = round(random.uniform(-0.5, 0.5), 2)
    w2 = round(random.uniform(-0.5, 0.5), 2)
    W = [w1, w2]
    X = [x1, x2]
    print("Initial weights:"+ str(W))
    print('x1\tx2\tYd\tY\te\tw1\tw2')
    # Iterate over epochs
    for i in range(1, n_epoch+1):
        print("Epoch:" + str(i))        
        # Loop through inputs in each epoch
        for p in range(0, len(x1)):
            # Activation - calculate actual output Y using step function
            Y = 0
            activationfn = np.dot(W, X)
            if activationfn[p] >= 0:
                Y =1
            # Calculate error
            e = Yd[p]-Y
            # Weight training
            w1 = w1 + alpha*x1[p]*e
            w2 = w2 + alpha*x2[p]*e
            print(str(x1[p])+'\t'+str(x2[p])+'\t'+str(Yd[p])+'\t' +str(Y)+'\t'
                +str(e)+'\t' +str(round(w1, 1))+'\t'+str(round(w2, 1)))

perceptAND(x1, x2, Yd, theta, alpha, n_epoch)
