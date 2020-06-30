import time

import tensorflow as tf
import numpy as np
import scipy.stats as stats
from utilities.ode_solver import tpbvp_hjb_solve_warm_start


class HJBValueNetwork:

    def __init__(self, problem, scaling, config, parameters=None):
        self.lb = scaling['lb']
        self.ub = scaling['ub']
        self.A_lb = scaling['A_lb']
        self.A_ub = scaling['A_ub']
        self.U_lb = scaling['U_lb']
        self.U_ub = scaling['U_ub']
        self.V_min = scaling['V_min']
        self.V_max = scaling['V_max']

        self.problem = problem
        self.config = config

        N_states = problem.N_states
        N_controls = problem.N_controls
        self.t1 = config.t1

        self.weights, self.biases = self.initialize_net(config.layers, parameters)

        self.t_tf = tf.placeholder(tf.float32, shape=(1, None))
        self.X_tf = tf.placeholder(tf.float32, shape=(N_states, None))

        V_pred_scaled, self.V_pred = self.make_eval_graph(self.t_tf, self.X_tf)
        self.dVdX = tf.gradients(self.V_pred, self.X_tf)[0]
        self.U = self.problem.make_U_NN(self.dVdX)
        dVdX_scaled = 2.0 * (self.dVdX - self.A_lb) / (self.A_ub - self.A_lb) - 1.0
        U_scaled = 2.0 * (self.U - self.U_lb) / (self.U_ub - self.U_lb) - 1.0

        self.A_scaled_tf = tf.placeholder(tf.float32, shape=(N_states, None))
        self.U_scaled_tf = tf.placeholder(tf.float32, shape=(N_controls, None))
        self.V_scaled_tf = tf.placeholder(tf.float32, shape=(1, None))

        self.loss_V = tf.reduce_mean((V_pred_scaled - self.V_scaled_tf) ** 2)
        self.loss_A = tf.reduce_mean(
            tf.reduce_sum((dVdX_scaled - self.A_scaled_tf) ** 2, axis=0)
        )
        self.loss_U = tf.reduce_mean(
            tf.reduce_sum((U_scaled - self.U_scaled_tf) ** 2, axis=0)
        )

        self.A_tf = tf.placeholder(tf.float32, shape=(N_states, None))
        self.U_tf = tf.placeholder(tf.float32, shape=(N_controls, None))
        self.V_tf = tf.placeholder(tf.float32, shape=(1, None))

        self.MAE = tf.reduce_mean(tf.abs(self.V_pred - self.V_tf))
        self.grad_MRL2 = tf.reduce_mean(tf.sqrt(
            tf.reduce_sum((self.dVdX - self.A_tf) ** 2, axis=0) / (
                    0.01 + tf.sqrt(tf.reduce_sum(self.A_tf ** 2, axis=0))
            )
        ))
        self.ctrl_MRL2 = tf.reduce_mean(tf.sqrt(
            tf.reduce_sum((self.U - self.U_tf) ** 2, axis=0) / (
                    0.01 + tf.sqrt(tf.reduce_sum(self.U_tf ** 2, axis=0))
            )
        ))

        dVdX_norm = tf.sqrt(tf.reduce_sum(self.dVdX ** 2, axis=0))
        self.k_largest = tf.placeholder(tf.int32, ())
        self.largest_dVdX = tf.nn.top_k(dVdX_norm, k=self.k_largest, sorted=False)
        self.largest_V = tf.nn.top_k(self.V_pred, k=self.k_largest, sorted=False)

        self.sess = None

    def make_eval_graph(self, t, X):
        """Builds the NN computational graph."""

        # (N_states, ?) matrix of linearly rescaled input values
        V = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        V = tf.concat([V, 2.0 * t / self.t1 - 1.0], axis=0)
        # Hidden layers
        for l in range(len(self.weights) - 1):
            W = self.weights[l]
            b = self.biases[l]
            V = tf.tanh(tf.matmul(W, V) + b)
        # The last layer is linear -> it's outside the loop
        W = self.weights[-1]
        b = self.biases[-1]
        V = tf.matmul(W, V) + b

        V_descaled = (self.V_max - self.V_min) * (V + 1.) / 2. + self.V_min

        return V, V_descaled

    def predict_V(self, t, X):
        return self.sess.run(self.V_pred, {self.t_tf: t, self.X_tf: X})

    def predict_A(self, t, X):
        return self.sess.run(self.dVdX, {self.t_tf: t, self.X_tf: X})

    def get_largest_A(self, t, X, N):
        """ Return k-largest indexes of A"""
        return self.sess.run(self.largest_dVdX,
                             {
                                 self.k_largest: N,
                                 self.t_tf: t,
                                 self.X_tf: X
                             })[1]

    def get_largest_V(self, t, X, N):
        return self.sess.run(self.largest_V,
                             {
                                 self.k_largest: N,
                                 self.t_tf: t,
                                 self.X_tf: X
                             })[1]

    def eval_U(self, t, X):
        return self.sess.run(self.U, {self.t_tf: t, self.X_tf: X}).astype(np.float64)

    def bvp_guess(self, t, X, eval_U=False):
        feed_dict = {self.t_tf: t, self.X_tf: X}
        if eval_U:
            V, A, U = self.sess.run((self.V_pred, self.dVdX, self.U), feed_dict)
            return V, A, U.astype(np.float64)
        else:
            return self.sess.run((self.V_pred, self.dVdX), feed_dict)

    def train(self, train_data, val_data, options):

        # Prepare data
        train_data.update(
            {
                'U': self.problem.U_star(np.vstack((train_data['X'], train_data['A'])))
            }
        )
        train_data.update({
            'A_scaled': 2. * (train_data['A'] - self.A_lb) / (self.A_ub - self.A_lb) - 1.,
            'U_scaled': 2. * (train_data['U'] - self.U_lb) / (self.U_ub - self.U_lb) - 1.,
            'V_scaled': 2. * (train_data['V'] - self.V_min) / (self.V_max - self.V_min) - 1.
        })
        val_data = {
            self.t_tf: val_data.pop('t'),
            self.X_tf: val_data.pop('X'),
            self.V_tf: val_data.pop('V'),
            self.A_tf: val_data.pop('A'),
        }
        val_data.update({
            self.U_tf: self.problem.U_star(np.vstack((val_data[self.X_tf], val_data[self.A_tf])))
        })

        # Rounds
        max_rounds = options.pop('max_rounds', 1)

        # weights of losses
        weight_A = options.pop('weight_A', np.zeros(max_rounds))
        weight_U = options.pop('weight_U', np.zeros(max_rounds))

        assert len(weight_A) == max_rounds and len(weight_U) == max_rounds, "Number of loss weights must equal to " \
                                                                            "number of rounds "
        self.weight_A_tf = tf.placeholder(tf.float32, shape=())
        self.weight_U_tf = tf.placeholder(tf.float32, shape=())
        self.loss = self.loss_V
        if weight_A[0] >= 10.0 * np.finfo(float).eps:
            self.loss += self.weight_A_tf * self.loss_A
        if weight_U[0] >= 10.0 * np.finfo(float).eps:
            self.loss += self.weight_U_tf * self.loss_U

        # Optimizer
        self.grads_list = [None] * 3
        optimizer = None

        # Records
        train_err = []
        train_grad_err = []
        train_ctrl_err = []

        val_err = []
        val_grad_err = []
        val_ctrl_err = []

        round_iters = []

        errors_to_track = [train_err, train_grad_err, train_ctrl_err]
        fetches = [[self.MAE, self.grad_MRL2, self.ctrl_MRL2]]
        # later fetches will be passed to loss_callback as:
        # loss_callback(*fetches),
        # our signature of loss_callback is:
        # def loss_callback(fetches_);
        # it will then match: fetches_ = [self.MAE, self.grad_MRL2, self.ctrl_MRL2]

        # Batch size, maximal batch size, number of candidate initial states, batch size growth factor
        self.Ns = options.pop('batch_size', train_data['X'].shape[1])
        Ns_max = options.pop('max_batch_size', 32768)
        Ns_cand = options.pop('Ns_cand', 2)
        Ns_C = options.pop('Ns_C', 2)

        BFGS_opts = options.pop('BFGS_opts', {})

        for round in range(1, max_rounds + 1):
            # Prepare data ---------------------------------------------------------------------------------------------
            if self.Ns > train_data['X'].shape[1]:
                new_data = self.generate_data(self.Ns - train_data['X'].shape[1], Ns_cand=Ns_cand)
                train_data.update({
                    't': np.hstack((train_data['t'], new_data['t'])),
                    'X': np.hstack((train_data['X'], new_data['X'])),
                    'A': np.hstack((train_data['A'], new_data['A'])),
                    'U': np.hstack((train_data['U'], new_data['U'])),
                    'V': np.hstack((train_data['V'], new_data['V'])),
                    'A_scaled': np.hstack((train_data['A_scaled'], new_data['A_scaled'])),
                    'U_scaled': np.hstack((train_data['U_scaled'], new_data['U_scaled'])),
                    'V_scaled': np.hstack((train_data['V_scaled'], new_data['V_scaled']))
                })
            self.Ns = np.minimum(self.Ns, Ns_max)

            print('Optimization round', round, ':')
            print('Batch size = %d, gradient weight = %1.1e, control weight = %1.1e'
                  % (self.Ns, weight_A[round - 1], weight_U[round - 1]))

            # select Ns samples to train
            idx = np.random.choice(
                train_data['X'].shape[1], self.Ns, replace=False
            )
            tf_dict = {
                self.t_tf: train_data['t'][:, idx],
                self.X_tf: train_data['X'][:, idx],
                self.A_tf: train_data['A'][:, idx],
                self.U_tf: train_data['U'][:, idx],
                self.V_tf: train_data['V'][:, idx],
                self.A_scaled_tf: train_data['A_scaled'][:, idx],
                self.U_scaled_tf: train_data['U_scaled'][:, idx],
                self.V_scaled_tf: train_data['V_scaled'][:, idx],
                self.weight_A_tf: weight_A[round - 1],
                self.weight_U_tf: weight_U[round - 1]
            }

            # Run one round --------------------------------------------------------------------------------------------
            _BFGS_opts = {}
            for key in BFGS_opts.keys():
                _BFGS_opts[key] = BFGS_opts[key][round - 1]

            optimizer = self._train_L_BFGS_B(tf_dict,
                                             optimizer=optimizer,
                                             error_to_track=errors_to_track,
                                             fetches=fetches,
                                             options=_BFGS_opts
                                             )

            # Re-Calculate training losses and validation metrics  -----------------------------------------------------
            loss_V, loss_A, loss_U = self.sess.run(
                (self.loss_V, self.loss_A, self.loss_U), tf_dict)
            print('')
            print('loss_V = %1.1e, loss_A = %1.1e, loss_U = %1.1e' % (loss_V, loss_A, loss_U))

            round_iters.append(len(train_err))

            val_errs = self.sess.run(
                (self.MAE, self.grad_MRL2, self.ctrl_MRL2), val_data)

            val_err.append(val_errs[0])
            val_grad_err.append(val_errs[1])
            val_ctrl_err.append(val_errs[2])

            print('')
            print('Training MAE error = %1.1e' % (train_err[-1]))
            print('Validation MAE error = %1.1e' % (val_err[-1]))
            print('Training grad. MRL2 error = %1.1e' % (train_grad_err[-1]))
            print('Validation grad. MRL2 error = %1.1e' % (val_grad_err[-1]))
            print('Training ctrl. MRL2 error = %1.1e' % (train_ctrl_err[-1]))
            print('Validation ctrl. MRL2 error = %1.1e' % (val_ctrl_err[-1]))

            self.Ns *= Ns_C

        errors = (np.array(train_err), np.array(train_grad_err),
                  np.array(train_ctrl_err), np.array(val_err),
                  np.array(val_grad_err), np.array(val_ctrl_err))

        return round_iters, errors

    def _train_L_BFGS_B(self,
                        tf_dict,
                        optimizer=None,
                        error_to_track=None,
                        fetches=None,
                        options=None):
        error_to_track = error_to_track or []
        fetches = fetches or []
        options = options or {}

        from utilities.optimize import ScipyOptimizerInterface

        if optimizer is None:
            default_opts = {'maxcor': 15, 'ftol': 1e-11, 'gtol': 1e-06,
                            'iprint': 95, 'maxfun': 100000, 'maxiter': 100000}
            optimizer = ScipyOptimizerInterface(self.loss,
                                                grads_list=self.grads_list,
                                                options={**default_opts, **options})
            self.grads_list = optimizer._grads_list
            self.packed_loss_grad = optimizer._packed_loss_grad

        def callback(fetches_):
            for error_list, fetch in zip(error_to_track, fetches_):
                error_list.append(fetch)

        self.run_initializer()
        optimizer.minimize(self.sess, feed_dict=tf_dict, fetches=fetches, loss_callback=callback)
        return optimizer

    def generate_data(self, Nd, Ns_cand):
        """
        generate Nd samples, each sample has initial state from Ns_cand candidates
        """
        import warnings
        np.seterr(over='warn', divide='warn', invalid='warn')
        warnings.filterwarnings('error')
        print("Generating data.")

        N_states = self.problem.N_states
        t_OUT = np.empty((1, 0))
        X_OUT = np.empty((N_states, 0))
        A_OUT = np.empty((N_states, 0))
        V_OUT = np.empty((1, 0))

        N_tries = 0
        N_sol = 0
        start_time = time.time()
        while X_OUT.shape[1] < Nd:
            X0 = self.sample_X0(Ns_cand, mode=self.config.data_mode['sample'])
            X0 = self.choose_state(X0, mode=self.config.data_mode['evaluate'])
            bc = self.problem.make_bc(X0)
            try:
                N_tries += 1
                status, t, X_aug = tpbvp_hjb_solve_warm_start(
                    self.problem.dynamics, self.eval_U, X0, [0., self.t1], self.config.ODE_solver,
                    self.bvp_guess,
                    self.problem.aug_dynamics, bc, self.config.data_tol, self.config.max_nodes
                )
                if not status:
                    warnings.warn(Warning())
                    N_sol += 1
                V = X_aug[-1:] + self.problem.terminal_cost(X_aug[:N_states, -1])
                t_OUT = np.hstack((t_OUT, t.reshape(1, -1)))
                X_OUT = np.hstack((X_OUT, X_aug[:N_states]))
                A_OUT = np.hstack((A_OUT, X_aug[N_states:2 * N_states]))
                V_OUT = np.hstack((V_OUT, V))
            except Warning:
                pass

        warnings.resetwarnings()
        print("Generated %d data from %d (of %d tries) BVP solutions in %.1f sec" %
              (X_OUT.shape[1], N_sol, N_tries, time.time() - start_time))
        data = {'t': t_OUT, 'X': X_OUT, 'A': A_OUT, 'V': V_OUT,
                'U': self.problem.U_star(np.vstack((X_OUT, A_OUT)))
                }
        data.update({
            'A_scaled': 2. * (data['A'] - self.A_lb) / (self.A_ub - self.A_lb) - 1.,
            'U_scaled': 2. * (data['U'] - self.U_lb) / (self.U_ub - self.U_lb) - 1.,
            'V_scaled': 2. * (data['V'] - self.V_min) / (self.V_max - self.V_min) - 1.
        })
        return data

    def sample_X0(self, n, mode='arbitrary', **kwargs):
        print("Sampling initial state using method %s" % mode)
        if mode == 'radius':
            radius = kwargs.get('radius', 1)
            X0 = np.random.rand(self.problem.N_states, n)
            X0 = X0 / np.linalg.norm(X0, axis=0)
            X0 = radius * X0
        else:
            X0 = (self.ub - self.lb) * np.random.rand(self.problem.N_states, n) + self.lb

        return X0

    def choose_state(self, x, mode='dVdX'):
        print("Choosing initial state using %s" % mode)
        if mode == 'V':
            idx = self.get_largest_V(np.zeros((1, x.shape[1])), x, 1)
        else:
            idx = self.get_largest_A(np.zeros((1, x.shape[1])), x, 1)

        return x[:, idx[0]]

    def initialize_net(self, layers, parameters):
        weights, biases = [], []
        if parameters is None:
            def xavier_init(size_in, size_out):
                # Initializes a single set of weights for layer (l) from layer (l-1).
                # Weights are picked randomly from a truncated normal distribution
                std = np.sqrt(6. / (size_in + size_out))
                init = stats.truncnorm.rvs(-3.0 * std, 3.0 * std,
                                           scale=std, size=(size_out, size_in))
                return tf.Variable(init, dtype=tf.float32)

            for l in range(len(layers) - 1):
                weights.append(xavier_init(layers[l], layers[l + 1]))
                biases.append(tf.Variable(tf.zeros((layers[l + 1], 1), dtype=tf.float32)))
        else:
            for l in range(len(parameters['weights'])):
                weights.append(tf.Variable(parameters['weights'][l], dtype=tf.float32))
                biases.append(tf.Variable(parameters['biases'][l], dtype=tf.float32))

        return weights, biases

    def run_initializer(self):
        if self.sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

    def export_model(self):
        """Returns a list of weights and biases to save model parameters."""
        weights = np.empty((len(self.weights),), dtype=object)
        biases = np.empty((len(self.biases),), dtype=object)

        for l in range(len(self.weights)):
            weights[l], biases[l] = self.sess.run(
                (self.weights[l], self.biases[l])
            )
        model_dict = {
            'lb': self.lb, 'ub': self.ub,
            'A_lb': self.A_lb, 'A_ub': self.A_ub,
            'U_lb': self.U_lb, 'U_ub': self.U_ub,
            'V_min': self.V_min, 'V_max': self.V_max
        }
        model_dict.update({
            'weights': weights, 'biases': biases
        })

        return model_dict
