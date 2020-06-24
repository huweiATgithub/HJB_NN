import time
import numpy as np
import scipy.io
from models.problem import ProblemPrototype, ConfigPrototype
from utilities.ode_solver import tpbvp_HJB_solve


def generate_data_time_march(problem: ProblemPrototype, config: ConfigPrototype,
                             X0_sample=None, initial_tol=1e-1,
                             save_path=None):
    X0s = X0_sample
    if X0s is None:
        X0s = problem.sample_X0(config.Ns['train'])
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
    while N_sol < Ns:
        X0 = X0s[:, N_sol]
        X0_aug_guess = np.vstack((X0.reshape(-1, 1), np.zeros(N_states + 1, 1)))
        bc = problem.make_bc(X0)

        start_time = time.time()

        status, t, X_sol = tpbvp_HJB_solve(problem.aug_dynamics, bc, X0_aug_guess,
                                           config.tseq, initial_tol, config.data_tol, config.max_nodes)
        end_time = time.time()
        time_elapsed = end_time - start_time
        if status:
            # as we don't consider terminal cost when solving HJB, we need to add it back to data
            V = X_sol[-1:] + problem.terminal_cost(X_sol[:N_states, -1])
            t_OUT = np.hstack((t_OUT, t.reshape(1, -1)))
            X_OUT = np.hstack((X_OUT, X_sol[:N_states]))
            A_OUT = np.hstack((A_OUT, X_sol[N_states:2 * N_states]))
            V_OUT = np.hstack((V_OUT, V))
            N_sol += 1
            N_success += 1
            sol_time.append(time_elapsed)
        else:
            if X0_sample is None:
                N_sol += 1
            else:
                X0s[:, N_sol] = problem.sample_X0(1)
            N_fail += 1
            fail_time.append(time_elapsed)

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

    if save_path:
        save_dict = {
            't': t_OUT,
            'X': X_OUT,
            'A': A_OUT,
            'V': V_OUT
        }
        scipy.io.savemat(save_path, save_dict)




