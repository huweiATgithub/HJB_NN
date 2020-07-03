import time
import warnings
from typing import List

import numpy as np
import scipy.io
from models.problem import ProblemPrototype, ConfigPrototype
from utilities.ode_solver import tpbvp_hjb_solve_time_march
from networks.neural_networks import HJBValueNetwork


def generate_data_time_march(problem: ProblemPrototype, config: ConfigPrototype,
                             X0_sample=None, initial_tol=1e-1,
                             save_path=None):
    X0s = X0_sample
    if isinstance(X0s, str):
        if X0s == 'train':
            X0s = problem.sample_X0(config.Ns['train'])
        elif X0s == 'val':
            X0s = problem.sample_X0(config.Ns['val'])
        else:
            return

    N_states, Ns = X0s.shape

    # for storing data
    t_OUT = np.empty((1, 0))
    X_OUT = np.empty((N_states, 0))
    A_OUT = np.empty((N_states, 0))
    V_OUT = np.empty((1, 0))

    sol_time = []
    fail_time = []
    N_sol = 0
    N_success = 0
    N_fail = 0

    # Also catch float point warning
    np.seterr(over='warn', divide='warn', invalid='warn')
    warnings.filterwarnings('error')

    while N_sol < Ns:
        print('Solving BVP #', N_sol + 1, 'of', Ns, '...', end='\n')
        X0 = X0s[:, N_sol]
        X0_aug_guess = np.vstack((X0.reshape(-1, 1), np.zeros((N_states + 1, 1))))
        bc = problem.make_bc(X0)

        start_time = time.time()
        try:
            status, t, X_sol = tpbvp_hjb_solve_time_march(problem.aug_dynamics, bc, X0_aug_guess,
                                                          config.tseq, initial_tol, config.data_tol, config.max_nodes)
            if not status:
                warnings.warn(Warning())

            # as we don't consider terminal cost when solving HJB, we need to add it back to data
            V = X_sol[-1:] + problem.terminal_cost(X_sol[:N_states, -1])
            t_OUT = np.hstack((t_OUT, t.reshape(1, -1)))
            X_OUT = np.hstack((X_OUT, X_sol[:N_states]))
            A_OUT = np.hstack((A_OUT, X_sol[N_states:2 * N_states]))
            V_OUT = np.hstack((V_OUT, V))
            N_sol += 1
            N_success += 1
            sol_time.append(time.time() - start_time)
        except Warning:
            if isinstance(X0_sample, str):
                X0s[:, N_sol] = problem.sample_X0(1)
            else:
                N_sol += 1
            N_fail += 1
            fail_time.append(time.time() - start_time)

    warnings.resetwarnings()

    sol_time = np.sum(sol_time)
    fail_time = np.sum(fail_time)
    print('')
    print(N_success, '/', N_success + N_fail, 'successful solution attempts:')
    print('Average solution time: %1.1f' % (sol_time / N_success), 'sec')
    print('Total solution time: %1.1f' % sol_time, 'sec')
    if N_fail >= 1:
        print('')
        print('Average failure time: %1.1f' % (fail_time / N_fail), 'sec')
        print('Total failure time: %1.1f' % fail_time, 'sec')
        print('Total working time: %1.1f' % (sol_time + fail_time), 'sec')

    print('')
    print('Total data generated:', X_OUT.shape[1])
    print('')

    U = problem.U_star(np.vstack((X_OUT, A_OUT)))
    save_dict = {
        't': t_OUT,
        'X': X_OUT,
        'A': A_OUT,
        'V': V_OUT,
        'U': U,
    }
    if save_path:
        scipy.io.savemat(save_path, save_dict)
    else:
        return save_dict


def generate_data_ensembles(master_model: HJBValueNetwork, models: List[HJBValueNetwork],
                            n_candidate, num_samples=1, sample_mode='arbitrary'):
    N_states = master_model.problem.N_states
    t_OUT = np.empty((1, 0))
    X_OUT = np.empty((N_states, 0))
    A_OUT = np.empty((N_states, 0))
    V_OUT = np.empty((1, 0))

    start_time = time.time()
    print("Generating data:")
    n = 0
    n_tries = 0
    while n < num_samples:
        n_tries += 1
        X0 = master_model.sample_X0(n_candidate, mode=sample_mode)
        values = [master_model.predict_V(t=np.zeros((1, n_candidate)), X=X0)]
        values += [model.predict_V(t=np.zeros((1, n_candidate)), X=X0) for model in models]
        values_np = np.vstack(values)
        variance = np.var(values_np, axis=0, dtype=np.float64)
        idx = np.argmin(variance).item()
        X0 = X0[:, idx]
        status, (t, X, A, V) = master_model.generate_single_data(X0)
        if status:
            n += 1
            t_OUT = np.hstack((t_OUT, t))
            X_OUT = np.hstack((X_OUT, X))
            A_OUT = np.hstack((A_OUT, A))
            V_OUT = np.hstack((V_OUT, V))

    print("Generated %d data from %d (of %d tries) BVP solutions in %.1f sec" %
          (X_OUT.shape[1], n, n_tries, time.time() - start_time))

    data = {'t': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT,
            'U': master_model.problem.U_star(np.vstack((X_OUT, A_OUT)))
            }
    data.update({
        'A_scaled': 2. * (data['A'] - master_model.A_lb) / (master_model.A_ub - master_model.A_lb) - 1.,
        'U_scaled': 2. * (data['U'] - master_model.U_lb) / (master_model.U_ub - master_model.U_lb) - 1.,
        'V_scaled': 2. * (data['V'] - master_model.V_min) / (master_model.V_max - master_model.V_min) - 1.
    })
    return data



