import numpy as np


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def cost_mse(y, tx, w, lambda_):
    e = y - tx.dot(w)
    return 0.5 * np.mean(e * e)


def gradient_mse(y, tx, w, lambda_):
    e = y - tx.dot(w)
    gradient = -1 * tx.T.dot(e) / len(tx)
    return gradient


def cost_logistic(y, tx, w, lambda_):
    return np.squeeze(np.log(1 + np.exp(tx.dot(w))).sum() - y.T.dot(tx.dot(w)))


def gradient_logistic(y, tx, w, lambda_):
    # print(y.shape, tx.shape, w.shape)
    return tx.T.dot(1.0 / (1 + np.exp(-tx.dot(w))) - y)


def cost_reg_logistic(y, tx, w, lambda_):
    return cost_logistic(y, tx, w, lambda_) + lambda_  * np.squeeze(w.T.dot(w)) / 2


def gradient_reg_logistic(y, tx, w, lambda_):
    return gradient_logistic(y, tx, w, lambda_) + lambda_ * w


def gradient_descent(y, tx, initial_w, max_iters, gamma, batch_size, cost_f, gradient_f, lambda_ = 0):
    # print(y.shape, tx.shape, initial_w.shape)
    w = initial_w
    for i in range(max_iters):
        for yn, xn in batch_iter(y, tx, batch_size=batch_size, shuffle = (batch_size == 1)):
            gradient = gradient_f(yn, xn, w, lambda_)
            w = w - gamma * gradient
#             if i % 100 == 0:
#                 print("Current iteration={i}, loss={l}".format(i=i, l=cost_f(y, tx, w, lambda_)))
    return w, cost_f(y, tx, w, lambda_)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, len(y), cost_mse, gradient_mse)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 10, cost_mse, gradient_mse)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, len(y), cost_logistic, gradient_logistic)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, 10, cost_reg_logistic, gradient_reg_logistic, lambda_)


def solve_normal_equations(y, tx, lambda_):
    left = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    right = tx.T.dot(y)
    w = np.linalg.solve(left, right)
    loss = cost_mse(y, tx, w, lambda_)
    return w, loss


def least_squares(y, tx):
    return solve_normal_equations(y, tx, 0)


def ridge_regression(y, tx, lambda_):
    return solve_normal_equations(y, tx, lambda_)
