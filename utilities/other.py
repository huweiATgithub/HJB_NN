import numpy as np
from scipy.io import loadmat


def cheb(N):
    '''Build Chebyshev differentiation matrix.
    Uses algorithm on page 54 of Spectral Methods in MATLAB by Trefethen.'''
    theta = np.pi / N * np.arange(0, N+1)
    X_nodes = np.cos(theta)

    X = np.tile(X_nodes, (N+1, 1))
    X = X.T - X

    C = np.concatenate(([2.], np.ones(N-1), [2.]))
    C[1::2] = -C[1::2]
    C = np.outer(C, 1./C)

    D = C / (X + np.identity(N+1))
    D = D - np.diag(D.sum(axis=1))

    # Clenshaw-Curtis weights
    # Uses algorithm on page 128 of Spectral Methods in MATLAB
    w = np.empty_like(X_nodes)
    v = np.ones(N-1)
    for k in range(2, N, 2):
        v -= 2.*np.cos(k * theta[1:-1]) / (k**2 - 1)

    if N % 2 == 0:
        w[0] = 1./(N**2 - 1)
        v -= np.cos(N*theta[1:-1]) / (N**2 - 1)
    else:
        w[0] = 1./N**2

    w[-1] = w[0]
    w[1:-1] = 2.*v/N

    return X_nodes, D, w


def load_neural_network(model_path, return_stats=False):
    model_dict = loadmat(model_path)
    parameters = {'weights': model_dict['weights'][0],
                  'biases': model_dict['biases'][0]
                  }
    scaling = {'lb': model_dict['lb'], 'ub': model_dict['ub'],
               'A_lb': model_dict['A_lb'], 'A_ub': model_dict['A_ub'],
               'U_lb': model_dict['U_lb'], 'U_ub': model_dict['U_ub'],
               'V_min': model_dict['V_min'].flatten(),
               'V_max': model_dict['V_max'].flatten()}
    return parameters, scaling
