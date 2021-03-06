import numpy as np
from scipy.integrate import solve_ivp, cumtrapz
import abc


class ConfigPrototype:

    def __init__(self):
        self.tseq = None

    def build_layers(self, N_states, time_dependent, N_layers, N_neurons):
        layers = [N_states] + N_layers * [N_neurons] + [1]

        if time_dependent:
            layers[0] += 1

        return layers


class ProblemPrototype:

    def __init__(self):
        """Class defining the OCP, dynamics, and controllers."""
        pass

    def sample_X0(self, Ns):
        """Uniform sampling from the initial condition domain."""
        X0 = np.random.rand(self.N_states, Ns)
        X0 = (self.X0_ub - self.X0_lb) * X0 + self.X0_lb

        if Ns == 1:
            X0 = X0.flatten()
        return X0

    def U_star(self, X_aug):
        '''Optimal control as a function of the costate.'''
        raise NotImplementedError

    def make_U_NN(self, A):
        """Makes TensorFlow graph of optimal control with NN value gradient."""
        import tensorflow as tf

        raise NotImplementedError

    def make_LQR(self, F, G, Q, Rinv, P1):
        GRG = G @ Rinv @ G.T

        def riccati_ODE(t, p):
            P = p.reshape(F.shape)
            PF = - P @ F
            dPdt = PF.T + PF - Q + P @ GRG @ P
            return dPdt.flatten()

        SOL = solve_ivp(riccati_ODE, [self.t1, 0.], P1.flatten(),
                        dense_output=True, method='LSODA', rtol=1e-04)
        return SOL.sol

    def U_LQR(self, t, X):
        t = np.reshape(t, (1,))
        P = self.P(t).reshape(self.N_states, self.N_states)
        return self.RG @ P @ X

    def make_bc(self, X0_in):
        """Makes a function to evaluate the boundary conditions for a given
        initial condition.
        (terminal cost is zero so final condition on lambda is zero)"""

        def bc(X_aug_0, X_aug_T):
            return np.concatenate((X_aug_0[:self.N_states] - X0_in,
                                   X_aug_T[self.N_states:]))

        return bc

    def running_cost(self, X, U):
        raise NotImplementedError

    def terminal_cost(self, X):
        raise NotImplementedError

    def compute_cost(self, t, X, U):
        '''Computes the accumulated cost of a state-control trajectory as
        an approximation of V(t).'''
        L = self.running_cost(X, U)
        J = cumtrapz(L, t, initial=0.)
        return self.terminal_cost(X[:, -1]) + J[0, -1] + L[0, -1] - J

    def Hamiltonian(self, t, X_aug):
        U = self.U_star(X_aug)
        L = self.running_cost(X_aug[:self.N_states], U)

        F = self.aug_dynamics(t, X_aug)
        H = L + np.sum(X_aug[self.N_states:2 * self.N_states] * F[:self.N_states],
                       axis=0, keepdims=True)

        return H

    def dynamics(self, t, X, U_fun):
        """Evaluation of the dynamics at a single time instance for closed-loop
        ODE integration."""
        U = U_fun([[t]], X.reshape((-1, 1))).flatten()

        raise NotImplementedError

    def aug_dynamics(self, t, X_aug):
        """Evaluation of the augmented dynamics at a vector of time instances
        for solution of the two-point BVP."""

        # Optimal control as a function of the costate
        U = self.U_star(X_aug)

        raise NotImplementedError
