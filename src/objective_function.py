import os
import copy

import numpy as np
import scipy as sp
import pathos.pools as pp
import dill
import sympy

import admittance_matrix
import singleton



class OptimizingGeneratorParameters:
    """Wrapper around 4 parameters which we want to clarify.

    Unfortunately, it is not allowed to simply add or remove optimizing
    parameters (of a generator) from this class. If you want to add
    or remove some such parameters, it will require changing
    of some code in this file (you should pay attention to substitution
    of the fields to symbolic expressions). To sum up, this class
    is only for readability of the code and not for extensibility.

    Attributes:
        D_Ya (float): generator damping
        Ef_a (float): generator field voltage magnitude
        M_Ya (float): what is M in the papers?
        X_Ya (float): generator reactance (inductance)
    """

    def __init__(self, D_Ya, Ef_a, M_Ya, X_Ya):
        """Initializes 4 fields.

        This method requires prior values of optimizing parameters of
        a generator. Obtained 4-dimensional point will be treated as
        the starting point for an optimization routine. Values of prior
        parameters should be enough closed to the true parameters.
        """
        self.D_Ya = D_Ya
        self.Ef_a = Ef_a
        self.M_Ya = M_Ya
        self.X_Ya = X_Ya

    @property
    def as_array(self):
        """Returns generator parameters as a numpy.array."""
        return np.array([self.D_Ya, self.Ef_a, self.M_Ya, self.X_Ya])



class ResidualVector:
    """Wrapper for calculations of R (residual vector).

     Based on the paper, vector R is equal to (Mr, Mi, Pr, Pi)^T.
     This class stores 4 functions for further computing subvectors
     of the vector R at any given point on demand
     (see self.compute method).

    Attributes:
        _freq_data (class FreqData): data in frequency domain
        _vector (numpy.array): preallocated buffer for vector R
        _subvectors (dict): contains 4 keys:
            'Mr', 'Mi', 'Pr', 'Pi'. Every key matches to view
            and function of corresponding subvector of the vector R.
    """

    def __init__(self, freq_data):
        """Prepares for computing the residual vector at any point.

        Stores data in frequency domain, 4 views
        and 4 compiled functions (see sympy.lambdify)
        for further computing the vector R at any point.

        Args:
            freq_data (class FreqData): data in frequency domain
        """
        self._freq_data = freq_data

        self._vector = np.zeros(4 * len(freq_data.freqs))
        self._subvectors = dict()
        self._init_subvectors(self._get_sym_exprs())


    def _get_sym_exprs(self):
        # constructing symbolic expressions of Mr, Mi, Pr, Pi subvectors
        matrix_Y = admittance_matrix.AdmittanceMatrix().Ys
        Y11 = matrix_Y[0, 0]
        Y12 = matrix_Y[0, 1]
        Y21 = matrix_Y[1, 0]
        Y22 = matrix_Y[1, 1]

        Y11r, Y11i = sympy.re(Y11), sympy.im(Y11)
        Y12r, Y12i = sympy.re(Y12), sympy.im(Y12)
        Y21r, Y21i = sympy.re(Y21), sympy.im(Y21)
        Y22r, Y22i = sympy.re(Y22), sympy.im(Y22)

        Vm, Va, Im, Ia = sympy.symbols('Vm Va Im Ia')
        Vmr, Vmi = sympy.re(Vm), sympy.im(Vm)
        Var, Vai = sympy.re(Va), sympy.im(Va)
        Imr, Imi = sympy.re(Im), sympy.im(Im)
        Iar, Iai = sympy.re(Ia), sympy.im(Ia)

        return {
            'Mr': Imr - Y11r*Vmr + Y11i*Vmi - Y12r*Var + Y12i*Vai,
            'Mi': Imi - Y11i*Vmr - Y11r*Vmi - Y12i*Var - Y12r*Vai,
            'Pr': Iar - Y21r*Vmr + Y21i*Vmi - Y22r*Var + Y22i*Vai,
            'Pi': Iai - Y21i*Vmr - Y21r*Vmi - Y22i*Var - Y22r*Vai
        }


    def _init_subvectors(self, sym_exprs):
        subvector_size = len(self._freq_data.freqs)
        self._subvectors['Mr'] = {
            'begin': 0,
            'end': subvector_size
        }
        self._subvectors['Mi'] = {
            'begin': subvector_size,
            'end': 2 * subvector_size
        }
        self._subvectors['Pr'] = {
            'begin': 2 * subvector_size,
            'end': 3 * subvector_size
        }
        self._subvectors['Pi'] = {
            'begin': 3 * subvector_size,
            'end': 4 * subvector_size
        }

        # initializing functions for computing elements of subvectors
        Vm, Va, Im, Ia = sympy.symbols('Vm Va Im Ia')
        D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya Omega_a',
            real=True
        )
        for subvector_name, subvector_expr in sym_exprs.items():
            self._subvectors[subvector_name]['function'] = sympy.lambdify(
                args=[Vm, Va, Im, Ia, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
                expr=subvector_expr,
                modules='numexpr'
            )


    def compute(self, optimizing_gen_params):
        """Computes the residual vector at the given point.

        It evaluates the residual vector at the point
        specified by 'optimizing_gen_params'
        and returns a numpy.array containing (4K+4) numbers.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            vector_R (numpy.array of (4K+4) numbers): residual vector
                evaluated at the given point
        """
        subvector_size = len(self._freq_data.freqs)

        for subvector in self._subvectors.values():
            self._vector[subvector['begin']:subvector['end']] = (
                subvector['function'](
                    Vm=self._freq_data.Vm,
                    Va=self._freq_data.Va,
                    Im=self._freq_data.Im,
                    Ia=self._freq_data.Ia,
                    D_Ya=np.array([optimizing_gen_params.D_Ya
                                   for _ in range(subvector_size)]),
                    Ef_a=np.array([optimizing_gen_params.Ef_a
                                   for _ in range(subvector_size)]),
                    M_Ya=np.array([optimizing_gen_params.M_Ya
                                   for _ in range(subvector_size)]),
                    X_Ya=np.array([optimizing_gen_params.X_Ya
                                   for _ in range(subvector_size)]),
                    Omega_a=2.0 * np.pi * self._freq_data.freqs
                )
            )

        vector_R = self._vector  # no copying
        return vector_R



class CovarianceMatrix:
    """Wrapper for calculations of covariance matrix at any point.

    Attributes:
        _freqs (np.array): frequencies (points in frequency domain)
        _matrix (scipy.sparse.csr_matrix): preallocated buffer
        _blocks (dict): contains 12 keys:
            'NrNr', 'NrQr', 'NrQi', 'NiNi', 'NiQr', 'NiQi',
            'QrQr', 'QiQi', 'QrNr', 'QrNi', 'QiNr', 'QiNi'.
            Every key matches to block's position (its upper left
            and bottom right angles) and function
            of corresponding block of the covariance matrix.
    """

    def __init__(self, freq_data):
        """Prepares for computing the covariance matrix.

        Stores data in frequency domain, 12 positions of blocks
        and 12 compiled functions (see sympy.lambdify)
        for further computing the covariance matrix at any point.

        Args:
            freq_data (class FreqData): data in frequency domain
        """
        self._freqs = freq_data.freqs

        self._blocks = dict()
        self._init_blocks(self._get_sym_exprs(freq_data))

        self._matrix = None
        self._init_matrix()


    def _get_sym_exprs(self, freq_data):
        # constructing symbolic expressions of 12 blocks
        std_eps_Vm = freq_data.std_w_Vm
        std_eps_Va = freq_data.std_w_Va
        std_eps_Im = freq_data.std_w_Im
        std_eps_Ia = freq_data.std_w_Ia

        matrix_Y = admittance_matrix.AdmittanceMatrix().Ys
        Y11 = matrix_Y[0, 0]
        Y12 = matrix_Y[0, 1]
        Y21 = matrix_Y[1, 0]
        Y22 = matrix_Y[1, 1]

        Y11r, Y11i = sympy.re(Y11), sympy.im(Y11)
        Y12r, Y12i = sympy.re(Y12), sympy.im(Y12)
        Y21r, Y21i = sympy.re(Y21), sympy.im(Y21)
        Y22r, Y22i = sympy.re(Y22), sympy.im(Y22)

        sym_exprs = {
            'NrNr': (
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            'NrQr': (
                std_eps_Vm**2 * (Y11r*Y21r + Y11i*Y21i) +
                std_eps_Va**2 * (Y12r*Y22r + Y12i*Y22i)
            ),
            'NrQi': (
                std_eps_Vm**2 * (Y11r*Y21i - Y11i*Y21r) +
                std_eps_Va**2 * (Y12r*Y22i - Y12i*Y22r)
            ),
            'NiNi': (
                std_eps_Im**2
                + std_eps_Vm**2 * (Y11r**2 + Y11i**2)
                + std_eps_Va**2 * (Y12r**2 + Y12i**2)
            ),
            'NiQr': (
                std_eps_Vm**2 * (Y11i*Y21r - Y11r*Y21i) +
                std_eps_Va**2 * (Y12i*Y22r - Y12r*Y22i)
            ),
            'NiQi': (
                std_eps_Vm**2 * (Y11i*Y21i + Y11r*Y21r) +
                std_eps_Va**2 * (Y12i*Y22i + Y12r*Y22r)
            ),
            'QrQr': (
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21r**2 + Y21i**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            ),
            'QiQi': (
                std_eps_Ia**2
                + std_eps_Vm**2 * (Y21i**2 + Y21r**2)
                + std_eps_Va**2 * (Y22r**2 + Y22i**2)
            )
        }
        sym_exprs['QrNr'] = sym_exprs['NrQr']
        sym_exprs['QrNi'] = sym_exprs['NiQr']
        sym_exprs['QiNr'] = sym_exprs['NrQi']
        sym_exprs['QiNi'] = sym_exprs['NiQi']
        return sym_exprs


    def _init_blocks(self, sym_exprs):
        block_size = len(self._freqs)

        # blocks in the 1st line
        self._blocks['NrNr'] = {
            'upper_left': (0, 0),
            'bottom_right': (block_size, block_size)
        }
        self._blocks['NrQr'] = {
            'upper_left': (0, 2*block_size),
            'bottom_right': (block_size, 3*block_size)
        }

        self._blocks['NrQi'] = {
            'upper_left': (0, 3*block_size),
            'bottom_right': (block_size, 4*block_size)
        }

        # blocks in the 2nd line
        self._blocks['NiNi'] = {
            'upper_left': (block_size, block_size),
            'bottom_right': (2*block_size, 2*block_size)
        }
        self._blocks['NiQr'] = {
            'upper_left': (block_size, 2*block_size),
            'bottom_right': (2*block_size, 3*block_size)
        }
        self._blocks['NiQi'] = {
            'upper_left': (block_size, 3*block_size),
            'bottom_right': (2*block_size, 4*block_size)
        }

        # blocks in the 3rd line
        self._blocks['QrNr'] = {
            'upper_left': (2*block_size, 0),
            'bottom_right': (3*block_size, block_size)
        }
        self._blocks['QrNi'] = {
            'upper_left': (2*block_size, block_size),
            'bottom_right': (3*block_size, 2*block_size)
        }
        self._blocks['QrQr'] = {
            'upper_left': (2*block_size, 2*block_size),
            'bottom_right': (3*block_size, 3*block_size)
        }

        # blocks in the 4th line
        self._blocks['QiNr'] = {
            'upper_left': (3*block_size, 0),
            'bottom_right': (4*block_size, block_size)
        }
        self._blocks['QiNi'] = {
            'upper_left': (3*block_size, block_size),
            'bottom_right': (4*block_size, 2*block_size)
        }
        self._blocks['QiQi'] = {
            'upper_left': (3*block_size, 3*block_size),
            'bottom_right': (4*block_size, 4*block_size)
        }

        # initializing functions for computing elements of blocks
        D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya Omega_a',
            real=True
        )
        for block_name, block_expr in sym_exprs.items():
            self._blocks[block_name]['function'] = sympy.lambdify(
                args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
                expr=block_expr,
                modules='numexpr'
            )


    def _init_matrix(self):
        block_size = len(self._freqs)
        matrix = np.zeros((4 * block_size, 4 * block_size))

        for block in self._blocks.values():
            y_block_begin = block['upper_left'][0]
            y_block_end = block['bottom_right'][0]
            x_block_begin = block['upper_left'][1]
            x_block_end = block['bottom_right'][1]
            matrix[range(y_block_begin, y_block_end),
                   range(x_block_begin, x_block_end)] = np.ones(block_size)

        assert np.count_nonzero(matrix) == 12*block_size
        self._matrix = sp.sparse.csr_matrix(matrix)


    def compute(self, optimizing_gen_params):
        """Computes the covariance matrix at the given point.

        Computes the numerical value of the covariance matrix
        at the point specified by 'optimizing_gen_params'.
        The obtained sparse matrix will have sizes (4K+4)*(4K+4)
        and contain only numbers (not symbols).

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma_L (scipy.sparse.csr_matrix): the covariance matrix
                evaluated at the given point
        """
        block_size = len(self._freqs)

        for block in self._blocks.values():
            block_diag = block['function'](
                D_Ya=np.array([optimizing_gen_params.D_Ya
                               for _ in range(block_size)]),
                Ef_a=np.array([optimizing_gen_params.Ef_a
                               for _ in range(block_size)]),
                M_Ya=np.array([optimizing_gen_params.M_Ya
                               for _ in range(block_size)]),
                X_Ya=np.array([optimizing_gen_params.X_Ya
                               for _ in range(block_size)]),
                Omega_a=2.0 * np.pi * self._freqs
            )

            y_block_begin = block['upper_left'][0]
            y_block_end = block['bottom_right'][0]
            x_block_begin = block['upper_left'][1]
            x_block_end = block['bottom_right'][1]
            self._matrix[range(y_block_begin, y_block_end),
                         range(x_block_begin, x_block_end)] = block_diag

        gamma_L = self._matrix  # no copying
        return gamma_L



class ObjectiveFunction:
    """Wrapper for calculations of the objective function at any point.

    Attributes:
        _R (class ResidualVector): attribute to simplify
            calculations of vector R (residual vector)
            at any point (that is for any parameters)
        _gamma_L (class CovarianceMatrix): attribute to simplify
            calculations of gamma_L (covariance matrix)
            at any point (that is for any parameters)
        _gen_params_prior_mean (numpy.array):
            a starting point for an optimization routine
        _inv_gamma_g (numpy.array): inverted covariance matrix
            of prior generator parameters
    """

    def __init__(self, freq_data,
                 gen_params_prior_mean, gen_params_prior_std_dev):
        """Prepares for computing the objective function at any point.

        Stores data in frequency domain, prior mean
        of generator parameters, inverted diagonal covariance matrix
        of its parameters. Prepares the vector R (residual vector)
        and gamma_L (covariance matrix) for computing at any point.

        Args:
            freq_data (class FreqData): data in frequency domain
            gen_params_prior_mean (class OptimizingGeneratorParameters):
                prior mean of generator's parameters
                (we are uncertain in their values)
            gen_params_prior_std_dev (class OptimizingGeneratorParameters):
                standard deviations of generator's parameters
                (how much we are uncertain in their values)
        """
        self._R = ResidualVector(freq_data)
        self._gamma_L = CovarianceMatrix(freq_data)
        self._gen_params_prior_mean = gen_params_prior_mean.as_array
        self._inv_gamma_g = np.diag(
            1.0 / (
                (gen_params_prior_mean.as_array *
                 gen_params_prior_std_dev.as_array)**2
            )
        )  # multiplied by gen_params_prior_mean.as_array (not 1.0)

        dill.settings['recurse'] = True
        cpu_count = os.cpu_count()
        process_pool_size = (cpu_count - 1) if cpu_count > 1 else 1
        self._funcs = [copy.deepcopy(self) for _ in range(process_pool_size)]
        self._process_pool = pp.ProcessPool(process_pool_size)



    def compute(self, optimizing_gen_params):
        """Computes value of the objective function at the given point.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            func_value (numpy.float64) of the objective function
                at the given point
        """
        curr_delta_params = (
            optimizing_gen_params.as_array - self._gen_params_prior_mean
        )

        computed_R = self._R.compute(optimizing_gen_params)
        computed_gamma_L = self._gamma_L.compute(optimizing_gen_params)
        computed_inv_gamma_L_dot_R = sp.sparse.linalg.spsolve(
            computed_gamma_L, computed_R
        )

        func_value = (
            curr_delta_params @ self._inv_gamma_g @ curr_delta_params
            + computed_R @ computed_inv_gamma_L_dot_R
        )
        # print('### DEBUG: func_value =', func_value,
        #       'at the point =', optimizing_gen_params.as_array)
        return func_value


    def compute_from_array(self, optimizing_gen_params):
        """Computes the objective function at the given point(s).

        This method just calls self.compute method
        transforming the sole argument from numpy.array to an instance
        or numpy.array of instances of class OptimizingGeneratorParameters.
        It is necessary to have such method because optimizers
        want to give an instance of numpy.array as an argument.
        If optimizing_gen_params is a 2D numpy.array, computations
        will be performed in parallel mode.

        Args:
            optimizing_gen_params (numpy.array
                with shape (4, ) or (points_n, 4)):
                current values of optimizing generator parameters

        Returns:
            func_value(s) (numpy.float64 or
                numpy.array with shape (points_n, ))
                of the objective function at the given point(s)

        Note:
            Be cautious using this method! The order of parameters
            is extremely important!
        """
        if len(optimizing_gen_params.shape) == 1:
            return self.compute(OptimizingGeneratorParameters(
                D_Ya=optimizing_gen_params[0],
                Ef_a=optimizing_gen_params[1],
                M_Ya=optimizing_gen_params[2],
                X_Ya=optimizing_gen_params[3]
            ))

        assert len(optimizing_gen_params.shape) == 2
        points_n = len(optimizing_gen_params)
        optimizing_gen_params_arrays = np.array([
            OptimizingGeneratorParameters(
                D_Ya=optimizing_gen_params[i, 0],
                Ef_a=optimizing_gen_params[i, 1],
                M_Ya=optimizing_gen_params[i, 2],
                X_Ya=optimizing_gen_params[i, 3]
            ) for i in range(points_n)
        ])

        batches_n = len(self._funcs)
        batches = np.array_split(optimizing_gen_params_arrays, batches_n)
        return np.concatenate(self._process_pool.map(
            lambda func, batch: np.array([func.compute(x) for x in batch]),
            self._funcs, batches
        ))

