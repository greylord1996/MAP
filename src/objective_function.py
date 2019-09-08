"""Objective function to minimize.

To obtain posterior parameters of a dynamical system from prior parameters
it is necessary to minimize the objective function (construction of it
is based on Bayesian approach). Generally, only 'ObjectiveFunction'
class should be used outside this module because other classes only
help to construct the objective function.

"""


import os
import copy

import numpy as np
import scipy as sp
import pathos.pools as pp
import dill
import sympy


class ResidualVector:
    """Wrapper for calculations of residual vector (denoted as R).

     This class stores 2m (2*n_outputs, see the paper) functions
     for further computing subvectors of the residual vector
     at any given point (see the 'compute' method).

    Attributes:
        _freq_data (class FreqData): Data in frequency domain.
        _vector (numpy.array): Preallocated buffer for vector R.
        _exprs (list): compiled function to compute subvectors
    """

    def __init__(self, freq_data, admittance_matrix):
        """Prepare for computing the residual vector at any point.

        Store data in frequency domain and compiled functions (see
        sympy.lambdify) for further computing all subvectors of
        the vector R at any point.

        Args:
            freq_data (FreqData): Data in frequency domain.
            admittance_matrix (AdmittanceMatrix): Admittance matrix
                (denoted as Y) of a dynamical system.
        """
        self._freq_data = freq_data
        self._exprs = []
        self._vector = np.zeros(
            2 * len(freq_data.outputs) * len(freq_data.freqs)
        )

        Y = admittance_matrix.data
        m, n = Y.shape
        outputs = sympy.Matrix([sympy.symbols('y0:{}'.format(m))])
        inputs = sympy.Matrix([sympy.symbols('u0:{}'.format(n))])
        residuals = []
        for p in range(len(outputs)):
            residuals.append(
                sympy.re(outputs[p])
                - sympy.re(Y[p, :]).dot(sympy.re(inputs))
                + sympy.im(Y[p, :]).dot(sympy.im(inputs))
            )
            residuals.append(
                sympy.im(outputs[p])
                - sympy.im(Y[p, :]).dot(sympy.re(inputs))
                - sympy.re(Y[p, :]).dot(sympy.im(inputs))
            )

        sys_params = admittance_matrix.params
        omega = admittance_matrix.omega
        args = inputs.tolist()[0] + outputs.tolist()[0] + sys_params + [omega]
        for residual in residuals:
            self._exprs.append(sympy.lambdify(
                args=args,
                expr=residual,
                modules='numexpr'
            ))

    def compute(self, sys_params):
        """Compute the residual vector at the given point.

        Args:
            sys_params (numpy.ndarray): Parameters of a dynamical system.

        Returns:
            vector_R (numpy.array): Residual vector evaluated
                at the given point.
        """
        n_freqs = len(self._freq_data.freqs)
        assert self._freq_data.inputs.shape[1] == n_freqs
        assert self._freq_data.outputs.shape[1] == n_freqs
        repeated_sys_params = np.repeat(
            np.reshape(sys_params, (len(sys_params), 1)),
            repeats=n_freqs, axis=1
        )

        for i in range(len(self._exprs)):
            self._vector[i*n_freqs:(i+1)*n_freqs] = self._exprs[i](
                *self._freq_data.inputs,
                *self._freq_data.outputs,
                *repeated_sys_params,
                2.0 * np.pi * self._freq_data.freqs
            )

        vector_R = self._vector  # no copying
        return vector_R


class CovarianceMatrix:
    """Wrapper for calculations of covariance matrix (denoted as gamma_L).

    Attributes:
        _freqs (np.array): Frequencies in frequency domain.
        _matrix (scipy.sparse.csr_matrix): Preallocated buffer for gamma_L.
        _exprs (list): compiled function to compute blocks of gamma_L.
    """

    def __init__(self, freq_data, admittance_matrix):
        """Prepares for computing the covariance matrix.

        Store frequencies and compiled functions (see sympy.lambdify)
        for further computing all blocks of the gamma_L matrix
        at any given point (see the 'compute' method).

        Args:
            freq_data (FreqData): Data in frequency domain.
            admittance_matrix (AdmittanceMatrix): Admittance matrix
                (denoted as Y) of a dynamical system.
        """
        self._freqs = freq_data.freqs

        self._exprs = []
        self._init_exprs(freq_data, admittance_matrix)

        self._matrix = None
        self._init_matrix(admittance_matrix.data.shape[0])

    def _init_exprs(self, freq_data, admittance_matrix):
        Y = admittance_matrix.data
        n_outputs = admittance_matrix.data.shape[0]
        inputs_vars = freq_data.input_std_devs**2
        outputs_vars = freq_data.output_std_devs**2

        blocks = [
            [None for _ in range(2*n_outputs)]
            for _ in range(2*n_outputs)
        ]
        for p in range(n_outputs):
            for q in range(n_outputs):
                blocks[2*p][2*q] = (
                    (outputs_vars[p] if p == q else 0.0)
                    + sum(
                        sympy.Matrix([inputs_vars])
                        .multiply_elementwise(sympy.re(Y[p, :]))
                        .multiply_elementwise(sympy.re(Y[q, :]))
                    )
                    + sum(
                        sympy.Matrix([inputs_vars])
                        .multiply_elementwise(sympy.im(Y[p, :]))
                        .multiply_elementwise(sympy.im(Y[q, :]))
                    )
                )
                blocks[2*p][2*q + 1] = (
                    sum(
                        sympy.Matrix([inputs_vars])
                        .multiply_elementwise(sympy.re(Y[p, :]))
                        .multiply_elementwise(sympy.im(Y[q, :]))
                    )
                    - sum(
                        sympy.Matrix([inputs_vars])
                        .multiply_elementwise(sympy.im(Y[p, :]))
                        .multiply_elementwise(sympy.re(Y[q, :]))
                    )
                )
                blocks[2*p + 1][2*q] = -blocks[2*p][2*q + 1]
                blocks[2*p + 1][2*q + 1] = blocks[2*p][2*q]

        sys_params = admittance_matrix.params
        omega = admittance_matrix.omega
        for row in blocks:
            lambdified_row = []
            for expr in row:
                lambdified_row.append(sympy.lambdify(
                    args=(sys_params + [omega]),
                    expr=expr,
                    modules='numexpr'
                ))
            self._exprs.append(lambdified_row)

    def _init_matrix(self, n_outputs):
        # Prepare sparse matrix
        block_size = len(self._freqs)
        cov_matrix_size = 2 * n_outputs * block_size
        matrix = np.zeros((cov_matrix_size, cov_matrix_size))

        for row_idx in range(len(self._exprs)):
            for column_idx in range(len(self._exprs[0])):
                y_block_begin = row_idx * block_size
                y_block_end = (row_idx + 1) * block_size
                x_block_begin = column_idx * block_size
                x_block_end = (column_idx + 1) * block_size
                block_diag = np.ones(block_size)
                matrix[range(y_block_begin, y_block_end),
                       range(x_block_begin, x_block_end)] = block_diag

        assert np.count_nonzero(matrix) == (4 * n_outputs**2) * block_size
        self._matrix = sp.sparse.csr_matrix(matrix)

    def compute(self, sys_params):
        """Compute the covariance matrix at the given point.

        Args:
            sys_params (numpy.ndarray): Parameters of a dynamical system.

        Returns:
            gamma_L (scipy.sparse.csr_matrix): the covariance matrix
                evaluated at the given point
        """
        block_size = len(self._freqs)
        n_freqs = len(self._freqs)
        repeated_sys_params = np.repeat(
            np.reshape(sys_params, (len(sys_params), 1)),
            repeats=n_freqs, axis=1
        )

        for row_idx in range(len(self._exprs)):
            for column_idx in range(len(self._exprs[0])):
                block_diag = self._exprs[row_idx][column_idx](
                    *repeated_sys_params,
                    2.0 * np.pi * self._freqs
                )
                y_block_begin = row_idx * block_size
                y_block_end = (row_idx + 1) * block_size
                x_block_begin = column_idx * block_size
                x_block_end = (column_idx + 1) * block_size
                self._matrix[range(y_block_begin, y_block_end),
                             range(x_block_begin, x_block_end)] = block_diag

        gamma_L = self._matrix  # no copying
        return gamma_L


class ObjectiveFunction:
    """Wrapper for calculations of the objective function at any point.

    Attributes:
        _R (ResidualVector): Residual vector R.
        _gamma_L (CovarianceMatrix): Covariance matrix gamma_L.
        _prior_params (numpy.ndarray): Starting point to be passed
            to the optimization routine.
        _inv_gamma_g (numpy.ndarray): inverted diagonal covariance
            matrix of prior generator parameters.
    """

    def __init__(self, freq_data, admittance_matrix,
                 prior_params, prior_params_std):
        """Prepare the objective function to be computed at any point.

        Args:
            freq_data (FreqData): Data in frequency domain.
            admittance_matrix (AdmittanceMatrix): Admittance matrix
                (denoted as Y) of a dynamical system.
            prior_params (numpy.ndarray): Prior parameters of a system.
            prior_params_std (numpy.ndarray): Prior uncertainties in
                system parameters (see the 'perturb_params' function).
        """
        self._R = ResidualVector(freq_data, admittance_matrix)
        self._gamma_L = CovarianceMatrix(freq_data, admittance_matrix)
        self._prior_params = prior_params
        self._inv_gamma_g = np.diag(
            1.0 / ((prior_params*prior_params_std)**2)
        )

        dill.settings['recurse'] = True
        cpu_count = os.cpu_count()
        process_pool_size = (cpu_count - 1) if cpu_count > 1 else 1
        self._funcs = [copy.deepcopy(self) for _ in range(process_pool_size)]
        self._process_pool = pp.ProcessPool(process_pool_size)

    def _compute(self, sys_params):
        # compute value of the objective function at just one point
        curr_delta_params = sys_params - self._prior_params

        computed_R = self._R.compute(sys_params)
        computed_gamma_L = self._gamma_L.compute(sys_params)
        computed_inv_gamma_L_dot_R = sp.sparse.linalg.spsolve(
            computed_gamma_L, computed_R
        )

        return (
            curr_delta_params @ self._inv_gamma_g @ curr_delta_params
            + computed_R @ computed_inv_gamma_L_dot_R
            # + np.log(np.linalg.det(computed_gamma_L))
        )

    def compute(self, sys_params):
        """Compute the objective function at the given point(s).

        This method just calls self._compute method. If the argument is
        a 2D numpy.ndarray, computations will be performed in parallel
        mode.

        Note:
            Remember that the order of parameters is extremely important.

        Args:
            sys_params (numpy.ndarray): Parameters of a dynamical system.

        Returns:
            out (numpy.float64 or numpy.array): value(s) of the
                objective function evaluated at the given point(s)
        """
        if len(sys_params.shape) == 1:
            return self._compute(sys_params)

        assert len(sys_params.shape) == 2
        batches_n = len(self._funcs)
        batches = np.array_split(sys_params, batches_n)
        return np.concatenate(self._process_pool.map(
            lambda func, batch: np.array([func._compute(x) for x in batch]),
            self._funcs, batches
        ))

