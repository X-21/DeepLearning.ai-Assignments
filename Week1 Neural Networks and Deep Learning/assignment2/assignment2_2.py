import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

if __name__ == '__main__':
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    index = 5
    plt.ion()
    plt.imshow(train_set_x_orig[index])
    # plt.show()
    print("y = " + str(train_set_y[:, index]) + ", it's a '"
          + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")
    '''
    **Exercise:** Find the values for:
    - m_train (number of training examples)
    - m_test (number of test examples)
    - num_px (= height = width of a training image)
    Remember that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3). 
    For instance, you can access `m_train` by writing `train_set_x_orig.shape[0]`.
    '''
    # START CODE HERE # (≈ 3 lines of code)
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    # END CODE HERE #

    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x shape: " + str(train_set_x_orig.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x shape: " + str(test_set_x_orig.shape))
    print("test_set_y shape: " + str(test_set_y.shape))

    '''
    For convenience, you should now reshape images of shape (num_px, num_px, 3) 
    in a numpy-array of shape (num_px  ∗  num_px  ∗  3, 1). 
    After this, our training (and test) dataset is a numpy-array 
    where each column represents a flattened image. 
    There should be m_train (respectively m_test) columns.
    Exercise: Reshape the training and test data sets so that images of size (num_px, num_px, 3) are 
    flattened into single vectors of shape (num_px  ∗  num_px  ∗  3, 1).

    A trick when you want to flatten a matrix X of shape (a,b,c,d) 
    to a matrix X_flatten of shape (b ∗ c ∗ d, a) is to use:

    X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X
    '''
    # Reshape the training and test examples

    # START CODE HERE # (≈ 2 lines of code)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    # END CODE HERE #

    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

a = 1
