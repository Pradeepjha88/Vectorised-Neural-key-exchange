
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

#
# def hebbianSingle(i,j,W, X, sigma, tau1, tau2, l):
#     k = W[i,j] +(X[i,j]* tau1 * theta(sigma[i], tau1) * theta(tau1, tau2))
#     return np.clip(k,-l,l)
#
#
# def antiHebbianSingle(i,j,W, X, sigma, tau1, tau2, l):
#     k = W[i, j] - (X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2))
#     return np.clip(k, -l, l)
#
#
# def randomWalkSingle(i,j,W, X, sigma, tau1, tau2, l):
#     k = W[i, j] + (X[i, j] * theta(sigma[i], tau1) * theta(tau1, tau2))
#     return np.clip(k, -l, l)

def theta(t1, t2):
    return 1 if t1 == t2 else 0


def hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    # num_cores = multiprocessing.cpu_count()
    # W1 = Parallel(n_jobs=num_cores)(delayed(hebbianSingle)(i,j,W,X,sigma,tau1,tau2,l) for (i,j),_ in np.ndenumerate(W))
    # return np.asarray(W1).reshape((k,n))
    thet2 = theta(tau1, tau2)
    W2 = np.clip(np.asarray(W) + np.transpose(
        np.transpose(np.asarray(X)) * tau1 * np.where(np.equal(sigma, tau1), 1, 0) * thet2).reshape((k, n)), -l, l)
    return np.asarray(W2).reshape((k, n))


def hebbianSerial(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] += X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j], -l, l)
    return W


def anti_hebbian(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    # num_cores = multiprocessing.cpu_count()
    # W1 = Parallel(n_jobs=num_cores)(
    #     delayed(antiHebbianSingle)(i, j, W, X, sigma, tau1, tau2, l) for (i, j), _ in np.ndenumerate(W))
    # return np.asarray(W1).reshape((k, n))
    thet2 = theta(tau1, tau2)
    W2 = np.clip(np.asarray(W) - np.transpose(
        np.transpose(np.asarray(X)) * tau1 * np.where(np.equal(sigma, tau1), 1, 0) * thet2).reshape((k, n)), -l, l)
    return np.asarray(W2).reshape((k, n))


def antiHebbianSerial(W, X, sigma, tau1, tau2, l):
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] -= X[i, j] * tau1 * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j], -l, l)
    return W

def random_walk(W, X, sigma, tau1, tau2, l):
    k, n = W.shape
    # num_cores = multiprocessing.cpu_count()
    # W1 = Parallel(n_jobs=num_cores)(
    #     delayed(randomWalkSingle)(i, j, W, X, sigma, tau1, tau2, l) for (i, j), _ in np.ndenumerate(W))
    # return np.asarray(W1).reshape((k, n))
    thet2 = theta(tau1, tau2)
    W2 = np.clip(np.asarray(W) + np.transpose(
        np.transpose(np.asarray(X)) * np.where(np.equal(sigma, tau1), 1, 0) * thet2).reshape((k, n)), -l, l)
    return np.asarray(W2).reshape((k, n))

def randomWalkSerial(W, X, sigma, tau1, tau2, l):
    for (i, j), _ in np.ndenumerate(W):
        W[i, j] += X[i, j] * theta(sigma[i], tau1) * theta(tau1, tau2)
        W[i, j] = np.clip(W[i, j], -l, l)
    return W


class Machine:
    def __init__(self, k=3, n=4, l=6):
        self.k = k
        self.n = n
        self.l = l
        self.W = np.random.randint(-l, l + 1, [k, n])

    def get_output(self, X):
        k = self.k
        n = self.n
        W = self.W
        X = X.reshape([k, n])
        sigma = np.sign(np.sum(X * W, axis=1)) # Compute inner activation sigma Dimension:[K]
        tau = np.prod(sigma) # The final output
        self.X = X
        self.sigma = sigma
        self.tau = tau
        return tau

    def __call__(self, X):
        return self.get_output(X)

    def update(self, tau2, update_rule='hebbian',serial = True):
        X = self.X
        tau1 = self.tau
        sigma = self.sigma
        W = self.W
        l = self.l
        if (tau1 == tau2):
            if update_rule == 'hebbian':
                if serial:
                    W = hebbianSerial(W, X, sigma, tau1, tau2, l)
                else:
                    W = hebbian(W, X, sigma, tau1, tau2, l)
            elif update_rule == 'anti_hebbian':
                if serial:
                    W = antiHebbianSerial(W, X, sigma, tau1, tau2, l)
                else:
                    W = anti_hebbian(W, X, sigma, tau1, tau2, l)
            elif update_rule == 'random_walk':
                if serial:
                    W = randomWalkSerial(W, X, sigma, tau1, tau2, l)
                else:
                    W = random_walk(W, X, sigma, tau1, tau2, l)
            else:
                raise Exception("Invalid update rule. Valid update rules are: " +
                                "\'hebbian\', \'anti_hebbian\' and \'random_walk\'.")
        return W