# perceptOR

# Full name: Karesia Ramlal
# Student ID: 812000717
# Email: karesia.ramlal@gmail.com
# Course Code: COMP 6930


# Inputs for perceptOR
x1 = [0, 0, 1, 1] 
x2 = [0, 1, 0, 1]
Yd = [0, 1, 1, 1]

# Initialization with specified theta
theta = 0.3
alpha = 0.1
n_epoch = 5

def perceptOR(x1, x2, Yd, theta, alpha, n_epoch):
    # Initialization with specified weights
    w1 = 0.4
    w2 = -0.2
    print('x1\tx2\tYd\te\tw1\tw2')
    # Iterate over epochs
    for i in range(1, n_epoch+1):
        print("Epoch:" + str(i))        
        # Loop through inputs in each epoch
        for p in range(0, len(x1)):
            # Activation - calculate actual output Y using step function
            Y = 0
            activationfn = ((x1[p]*w1)+(x2[p]*w2)) - theta
            if activationfn >= 0:
                Y =1
            # Calculate error
            e = Yd[p]-Y
            # Weight training
            w1 = w1 + alpha*x1[p]*e
            w2 = w2 + alpha*x2[p]*e
            print(str(x1[p])+'\t'+str(x2[p])+'\t'+str(Yd[p])+'\t' +str(Y)+'\t'
                +str(e)+'\t' +str(round(w1, 1))+'\t'+str(round(w2, 1)))

perceptOR(x1, x2, Yd, theta, alpha, n_epoch)