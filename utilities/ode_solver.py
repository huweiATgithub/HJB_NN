import numpy as np
from scipy.integrate import solve_bvp


def tpbvp_HJB_solve(aug_dynamics: callable, bc: callable, x, time_marchs, initial_tol, tol, max_nodes):
    """
    :param max_nodes: maximal number of mesh nodes
    :param bc: boundary condition
    :param x: initial guess, shape N,1
    :param aug_dynamics: the dynamics
    :param tol: tolerence in the whole BVP
    :param time_marchs: should be increasing sequence of real number; time_marchs[0,-1] = t0, T
    :param initial_tol: initial tolerance in solving one stage bvp, we half it every stage until it reach tol
    :return:
    """
    t = np.array([time_marchs[0]])
    x_guess = x.reshape(-1, 1)
    tolerance = initial_tol
    status = True
    for k in range(time_marchs.shape[0]-1):
        if tolerance >= 2 * tol:
            tolerance /= 2.
        if k == time_marchs.shape[0]-2:
            tolerance = tol

        t = np.concatenate((t, time_marchs[k+1:k+2]))
        x_guess = np.hstack((x_guess, x_guess[:, -1:]))  # use current value to be next guess

        SOL = solve_bvp(aug_dynamics, bc, t, x_guess, tol=tolerance, max_nodes=max_nodes, verbose=0)
        if not SOL.success:
            status = False
            print(SOL.message)
            break
        t = SOL.x
        x_guess = SOL.y

    return status, t, x_guess
