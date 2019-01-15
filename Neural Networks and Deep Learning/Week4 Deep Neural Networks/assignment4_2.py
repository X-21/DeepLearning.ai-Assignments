from dnn_app_utils_v2 import *
from assignment4_1 import \
    initialize_parameters, \
    linear_activation_forward, \
    compute_cost, \
    linear_activation_backward, \
    update_parameters
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    np.random.seed(1)

    # dataset
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    index = 7
    plt.figure()
    plt.imshow(train_x_orig[index])
    print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
    #plt.show()
    plt.close()

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("Number of training examples: " + str(m_train))
    print("Number of testing examples: " + str(m_test))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_x_orig shape: " + str(train_x_orig.shape))
    print("train_y shape: " + str(train_y.shape))
    print("test_x_orig shape: " + str(test_x_orig.shape))
    print("test_y shape: " + str(test_y.shape))

    '''
    As usual, you reshape and standardize the images before feeding them to the network. 
    '''
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(m_train, -1).T  # shape: 12288x209
    test_x_flatten = test_x_orig.reshape(m_test, -1).T  # shape: 12288x50
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))

    '''
    Two-layer neural network
    '''
    n_x = 12288  # num_px * num_px * 3
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)





    a = 1
