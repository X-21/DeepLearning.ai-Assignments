from testCases import *
import numpy as np
from matplotlib import pyplot as plt
import sklearn
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    cost = np.sum((1 / m) * (np.dot((-Y), np.log(A2).T) - np.dot((1 - Y), np.log(1 - A2).T)))

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


if __name__ == '__main__':
    np.random.seed(1)  # set a seed so that the results are consistent
    X, Y = load_planar_dataset()

    plt.figure()
    plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral)
    #plt.show()
    plt.close()

    # Lets first get a better sense of what our data is like.
    shape_X = X.shape
    shape_Y = Y.shape
    m = shape_X[1]  # training set size
    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('I have m = %d training examples!' % (m))

    '''
    Simple Logistic Regression
    '''
    # clf = sklearn.linear_model.LogisticRegressionCV()
    # clf.fit(X.T, Y.T)
    # # Plot the decision boundary for logistic regression
    # plt.figure()
    # plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    # plt.title("Logistic Regression")
    # plt.show()
    # plt.close()
    #
    # # Print accuracy
    # LR_predictions = clf.predict(X.T)
    # print('Accuracy of logistic regression: %d ' % float(
    #     (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
    #       '% ' + "(percentage of correctly labelled datapoints)")

    '''
    Neural Network model
    
    Reminder: The general methodology to build a Neural Network is to:
    1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
    2. Initialize the model's parameters
    3. Loop:
        - Implement forward propagation
        - Compute loss
        - Implement backward propagation to get the gradients
        - Update parameters (gradient descent)
    '''
    # Defining the neural network structure
    # - n_x: the size of the input layer
    # - n_h: the size of the hidden layer (set this to 4)
    # - n_y: the size of the output layer
    X_assess, Y_assess = layer_sizes_test_case()
    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))

    # Initialize the model's parameters
    n_x, n_h, n_y = initialize_parameters_test_case()
    parameters = initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    '''
    The Loop
    '''
    # Forwardpropagation
    X_assess, parameters = forward_propagation_test_case()
    A2, cache = forward_propagation(X_assess, parameters)
    print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

    # compute the value of the cost J
    A2_t, Y_assess, parameters = compute_cost_test_case()
    print("cost = " + str(compute_cost(A2_t, Y_assess, parameters)))

    # Backpropagation
    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
    grads = backward_propagation(parameters, cache, X_assess, Y_assess)
    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("db2 = " + str(grads["db2"]))

    a = 1
