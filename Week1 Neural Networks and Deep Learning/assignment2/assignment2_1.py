import numpy as np


def softmax(sm_x):
    x_exp = np.exp(sm_x)
    x_exp_sum = np.sum(x_exp, axis=1, keepdims=True)
    x_result = x_exp / x_exp_sum
    return x_result


def loss1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """

    # START CODE HERE # (≈ 1 line of code)
    loss = np.sum(np.abs(y - yhat))
    # END CODE HERE #

    return loss


def loss2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """

    # START CODE HERE # (≈ 1 line of code)
    loss = np.sum(np.abs(y - yhat) ** 2)
    # loss2 = np.dot((y - yhat), (y - yhat).T)
    # END CODE HERE #

    return loss


if __name__ == '__main__':
    # softmax function
    x = np.array([
        [9, 2, 5, 0, 0],
        [7, 5, 0, 0, 0]])
    print("softmax(x) = " + str(softmax(x)))

    # Implement the L1 and L2 loss functions

    y_hat = np.array([.9, 0.2, 0.1, .4, .9])
    y_ori = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(loss1(y_hat, y_ori)))
    print("L2 = " + str(loss2(y_hat, y_ori)))
