import numpy as np
import math
import timeit


"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""

def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):
    start = timeit.default_timer()
    n = 200
    N = x_train.shape[0]
    con_y_train = y_train.reshape(y_train.shape[0],1)
    losses = []
    for e in range(epoch):
        sum_loss = 0
        con_train = np.concatenate((x_train,con_y_train),axis=1)
        if shuffle:
            np.random.shuffle(con_train)
        shuffle_y_train = con_train[:,-1]
        shuffle_x_train = con_train[:,0:-1]
        for i in range(int(N/n)):
            new_x_train = shuffle_x_train[i*n:(i+1)*n]
            new_y_train = shuffle_y_train[i*n:(i+1)*n]
            loss,w1, w2, w3, w4, b1, b2, b3, b4 = four_nn(new_x_train,new_y_train,w1, w2, w3, w4, b1, b2, b3, b4, False)
            sum_loss += loss
        losses.append(sum_loss)
    stop = timeit.default_timer()
    print('Timer: ',stop - start)
    print("losses: ",losses)
    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):
    accuracy = 0
    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    number_per_class=[0] * num_classes
    correct_number_per_class= [0] * num_classes
    con_matrix = np.zeros(100).reshape((10,10))
    pred_label = four_nn(x_test,y_test,w1, w2, w3, w4, b1, b2, b3, b4, True)
    for i in range(pred_label.size):
        number_per_class[y_test[i]]+=1
        con_matrix[y_test[i]][int(pred_label[i])] += 1
        if pred_label[i] == y_test[i]:
            accuracy += 1
            correct_number_per_class[int(pred_label[i])]+=1
    avg_class_rate = accuracy/y_test.size
    class_rate_per_class= np.array(correct_number_per_class)/number_per_class
    con_matrix = (con_matrix.transpose()/number_per_class).T
    print(con_matrix)

    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(x_train, y_train, w1, w2, w3, w4, b1, b2, b3, b4,test):
    eta = 0.1
    pred_label = np.zeros(len(y_train))
    Z1, acache1 = affine_forward(x_train,w1,b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1,w2,b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2,w3,b3)
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3,w4,b4)
    if test == True:
        for i in range(F.shape[0]):
            pred_label[i] = np.argmax(F[i])
        return pred_label
    else:
        loss, dF = cross_entropy(F,y_train)
        dA3, dW4, dB4 = affine_backward(dF,acache4)
        dZ3 = relu_backward(dA3,rcache3)
        dA2, dW3, dB3 = affine_backward(dZ3,acache3)
        dZ2 = relu_backward(dA2,rcache2)
        dA1, dW2, dB2 = affine_backward(dZ2,acache2)
        dZ1 = relu_backward(dA1,rcache1)
        dX, dW1, dB1 = affine_backward(dZ1,acache1)
        w1 = w1 - eta*dW1
        w2 = w2 - eta*dW2
        w3 = w3 - eta*dW3
        w4 = w4 - eta*dW4
        b1 = b1 - eta*dB1
        b2 = b2 - eta*dB2
        b3 = b3 - eta*dB3
        b4 = b4 - eta*dB4
    
    return loss, w1, w2, w3, w4, b1, b2, b3, b4

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    cache = (A,W,b)
    Z_product = np.matmul(A,W)
    b_new = np.tile(b,(A.shape[0],1))
    Z = np.array(b_new)+np.array(Z_product)
    return Z, cache

def affine_backward(dZ, cache):
    new_w = cache[1].transpose()
    dA = np.matmul(dZ,new_w)
    new_a = cache[0].transpose()
    dW = np.matmul(new_a,dZ)
    new_b = np.ones((1,dZ.shape[0]))
    dB = np.matmul(new_b,dZ)
    dB=dB.flatten()
    return dA, dW, dB

def relu_forward(Z):
    cache = Z.copy()
    Z_new = Z.copy()
    Z_new[Z_new<0] = 0
    A = Z_new
    return A, cache

def relu_backward(dA, cache):
    dZ = dA.copy()
    dZ[cache<=0] = 0
    return dZ

def cross_entropy(F, y):
    n_size = F.shape[0]
    sum1 = 0
    y_unique = np.unique(y)
    num_class = y_unique.shape[0]
    new_F = np.zeros((n_size,num_class))
    e_F = np.exp(F)
    sum2 = np.sum(e_F,axis=1)
    sum2_log=sum2.copy()
    for i in range(n_size):
        sum1 = sum1 + F[i][int(y[i])]
        new_F[i][int(y[i])] = 1
        sum2_log[i] = math.log(sum2[i],math.e)

    total_sum2_log=np.sum(sum2_log)
    d_e_F = e_F.T/(sum2)
    loss = -(1/n_size)*(sum1-total_sum2_log)
    dF = (new_F-d_e_F.T)/(-n_size)
    return loss, dF


