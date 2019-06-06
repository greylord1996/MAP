import numpy as np
import scipy as sp
import sympy

import admittance_matrix
import utils



class OptimizingGeneratorParameters:
    """Wrapper around 4 parameters of a generator which we want to clarify.

    Unfortunately, it is not allowed to simply add or remove optimizing
    parameters (of a generator) from this class. If you want add or remove
    some such parameters, it will require changing some code in this file
    (you should pay attention to substitution of the fields to
    symbolic expressions). To sum up, this class is only for readability
    of the code and not for extensibility.

    Attributes:
        D_Ya (float): generator damping
        Ef_a (float): generator field voltage magnitude
        M_Ya (float): what is M in the papers?
        X_Ya (float): generator reactance (inductance)
    """

    def __init__(self, D_Ya, Ef_a, M_Ya, X_Ya):
        """Inits all fields which represents a starting point for an optimizer.

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



def _construct_gen_params_arrays(optimizing_gen_params, freq_data_points_n):
    # constructing 4 arrays containing repeated values of optimizing_gen_params
    return {
        'D_Ya': np.array([
            optimizing_gen_params.D_Ya for _ in range(freq_data_points_n)
        ]),
        'Ef_a': np.array([
            optimizing_gen_params.Ef_a for _ in range(freq_data_points_n)
        ]),
        'M_Ya': np.array([
            optimizing_gen_params.M_Ya for _ in range(freq_data_points_n)
        ]),
        'X_Ya': np.array([
            optimizing_gen_params.X_Ya for _ in range(freq_data_points_n)
        ])
    }



# @utils.singleton
class ResidualVector:
    """Wrapper for calculations of R (residual vector).

     Based on the paper, vector R is equal to (Mr, Mi, Pr, Pi)^T.
     But the vector R is not stored. Instead of it this class stores
     only 4 functions for further computing and constructing the vector R
     at any given point on demand (see self.compute method).

    Attributes:
        _freq_data (class FreqData): data in frequency domain
        _elements (dict): contains 4 keys: 'Mr', 'Mi', 'Pr', 'Pi'.
            Every key matches to function for computing
            corresponding element of the vector R.
        _partial_derivatives (dict of dicts): contains 4 keys:
            'Mr', 'Mi', 'Pr', 'Pi'.
            Every key matches to dictionary holding 4 functions:
            'D_Ya': (function) for computing partial derivative at D_Ya
            'Ef_a': (function) for computing partial derivative at Ef_a
            'M_Ya': (function) for computing partial derivative at M_Ya
            'X_Ya': (function) for computing partial derivative at X_Ya

    Note:
        All attributes are private. Don't use them outside this class.
        Communicate with an instance of this class only via
        its public methods.
    """

    def __init__(self, freq_data):
        """Prepares for computing the residual vector and its gradient.

        Stores data in frequency domain and 4 compiled functions
        (see sympy.lambdify) for further computing and constructing
        the vector R at any 4-dimensional point (the number of parameters
        which we optimize is equal to 4).

        Args:
            freq_data (class FreqData): data in frequency domain
        """
        self._freq_data = freq_data

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

        sym_exprs = {
            'Mr': Imr - Y11r*Vmr + Y11i*Vmi - Y12r*Var + Y12i*Vai,
            'Mi': Imi - Y11i*Vmr - Y11r*Vmi - Y12i*Var - Y12r*Vai,
            'Pr': Iar - Y21r*Vmr + Y21i*Vmi - Y22r*Var + Y22i*Vai,
            'Pi': Iai - Y21i*Vmr - Y21r*Vmi - Y22i*Var - Y22r*Vai
        }

        self._elements = dict()
        self._init_elements(sym_exprs)

        self._partial_derivatives = dict()
        self._init_partial_derivatives(sym_exprs)


    def _init_elements(self, sym_exprs):
        Vm, Va, Im, Ia = sympy.symbols('Vm Va Im Ia')
        D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya Omega_a',
            real=True
        )
        for element_name, element_expr in sym_exprs.items():
            self._elements[element_name] = sympy.lambdify(
                args=[Vm, Va, Im, Ia, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
                expr=element_expr,
                modules='numpy'
            )


    def _init_partial_derivatives(self, sym_exprs):
        Vm, Va, Im, Ia = sympy.symbols('Vm Va Im Ia')
        D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya Omega_a',
            real=True
        )
        gen_params_dict = {
            'D_Ya': D_Ya, 'Ef_a': Ef_a, 'M_Ya': M_Ya, 'X_Ya': X_Ya
        }

        for element_name, element_expr in sym_exprs.items():
            self._partial_derivatives[element_name] = dict()
            for gen_param_name, gen_param in gen_params_dict.items():
                self._partial_derivatives[element_name][gen_param_name] = (
                    sympy.lambdify(
                        args=[Vm, Va, Im, Ia, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
                        expr=sympy.diff(element_expr, gen_param),
                        modules='numpy'
                    )
                )


    def _construct_vector_from_subvectors(self, subvectors):
        # constructing vector R from 4 subvectors: Mr, Mi, Pr, Pi
        return np.concatenate([
            subvectors['Mr'],
            subvectors['Mi'],
            subvectors['Pr'],
            subvectors['Pi']
        ])


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
                evaluated at the given 4-dimensional point (specified by
                the 'optimizing_gen_params' argument of this method)
        """
        optimizing_gen_params_arrays = _construct_gen_params_arrays(
            optimizing_gen_params=optimizing_gen_params,
            freq_data_points_n=len(self._freq_data.freqs)
        )

        vector_R_subvectors = dict()
        for subvector_name, subvector_function in self._elements.items():
            vector_R_subvectors[subvector_name] = subvector_function(
                Vm=self._freq_data.Vm,
                Va=self._freq_data.Va,
                Im=self._freq_data.Im,
                Ia=self._freq_data.Ia,
                D_Ya=optimizing_gen_params_arrays['D_Ya'],
                Ef_a=optimizing_gen_params_arrays['Ef_a'],
                M_Ya=optimizing_gen_params_arrays['M_Ya'],
                X_Ya=optimizing_gen_params_arrays['X_Ya'],
                Omega_a=2.0 * np.pi * self._freq_data.freqs
            )

        vector_R = self._construct_vector_from_subvectors(vector_R_subvectors)
        return vector_R


    def compute_partial_derivatives(self, optimizing_gen_params):
        """Computes partial derivatives of the residual vector.

        Each element of the residual vector depends on 9 quantities
        Vm, Va, Im, Ia, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a.
        This method constructs 4 vectors.
        The 1st vector consists of partial derivatives of R at D_Ya.
        The 2nd vector consists of partial derivatives of R at Ef_a.
        The 3rd vector consists of partial derivatives of R at M_Ya.
        The 4th vector consists of partial derivatives of R at X_Ya.
        Then it returns these 4 vectors in a dictionary.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            vector_R_partial_derivatives (dict): a dictionary with 4 keys:
                'D_Ya' (numpy.array): vector of partial derivatives at D_Ya
                'Ef_a' (numpy.array): vector of partial derivatives at Ef_a
                'M_Ya' (numpy.array): vector of partial derivatives at M_Ya
                'X_Ya' (numpy.array): vector of partial derivatives at X_Ya
        """
        optimizing_gen_params_arrays = _construct_gen_params_arrays(
            optimizing_gen_params=optimizing_gen_params,
            freq_data_points_n=len(self._freq_data.freqs)
        )

        vector_R_partial_derivatives = dict()
        for optimizing_gen_param_name in optimizing_gen_params_arrays.keys():
            partial_derivatives_subvectors = dict()
            for subvector_name, partial_derivatives_functions in (
                    self._partial_derivatives.items()):
                partial_derivatives_subvectors[subvector_name] = (
                    partial_derivatives_functions[optimizing_gen_param_name](
                        Vm=self._freq_data.Vm,
                        Va=self._freq_data.Va,
                        Im=self._freq_data.Im,
                        Ia=self._freq_data.Ia,
                        D_Ya=optimizing_gen_params_arrays['D_Ya'],
                        Ef_a=optimizing_gen_params_arrays['Ef_a'],
                        M_Ya=optimizing_gen_params_arrays['M_Ya'],
                        X_Ya=optimizing_gen_params_arrays['X_Ya'],
                        Omega_a=2.0 * np.pi * self._freq_data.freqs
                    )
                )

            vector_R_partial_derivatives[optimizing_gen_param_name] = (
                self._construct_vector_from_subvectors(
                    partial_derivatives_subvectors
                )
            )

        return vector_R_partial_derivatives



# @utils.singleton
class CovarianceMatrix:
    """Wrapper for calculations of covariance matrix at any point.

    Attributes:
        _freqs (np.array): frequencies in frequency domain
        _elements (dict): contains 12 keys:
            'NrNr', 'NrQr', 'NrQi', 'NiNi', 'NiQr', 'NiQi',
            'QrQr', 'QiQi', 'QrNr', 'QrNi', 'QiNr', 'QiNi'.
            Every key matches to function for computing
            corresponding element of the gamma_L matrix.
        _partial_derivatives (dict of dicts): contains 12 keys:
            'NrNr', 'NrQr', 'NrQi', 'NiNi', 'NiQr', 'NiQi',
            'QrQr', 'QiQi', 'QrNr', 'QrNi', 'QiNr', 'QiNi'.
            Every key matches to dictionary holding 4 functions:
            'D_Ya': (function) for computing partial derivative at D_Ya
            'Ef_a': (function) for computing partial derivative at Ef_a
            'M_Ya': (function) for computing partial derivative at M_Ya
            'X_Ya': (function) for computing partial derivative at X_Ya

    Note:
        All attributes are private. Don't use them outside this class.
        Communicate with an instance of this class only via
        its public methods.
    """

    def __init__(self, freq_data):
        """Prepares for computing the covariance matrix and its gradient.

        Stores data in frequency domain, 12 compiled functions
        (see sympy.lambdify) for further computing and constructing
        the gamma_L matrix at any 4-dimensional point
        (the number of parameters which we optimize is equal to 4)
        and 48 compiled functions for computing and constructing 4 matrices
        (each of the 4 matrices contains partial derivatives
        at D_Ya, Ef_a, M_Ya, X_Ya respectively).

        Args:
            freq_data (class FreqData): data in frequency domain
        """
        self._freqs = freq_data.freqs

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

        self._elements = dict()
        self._init_elements(sym_exprs)

        self._partial_derivatives = dict()
        self._init_partial_derivatives(sym_exprs)


    def _init_elements(self, sym_exprs):
        D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya Omega_a',
            real=True
        )
        for element_name, element_expr in sym_exprs.items():
            self._elements[element_name] = sympy.lambdify(
                args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
                expr=element_expr,
                modules='numpy'
            )


    def _init_partial_derivatives(self, sym_exprs):
        D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya Omega_a',
            real=True
        )
        gen_params_dict = {
            'D_Ya': D_Ya, 'Ef_a': Ef_a, 'M_Ya': M_Ya, 'X_Ya': X_Ya
        }

        for element_name, element_expr in sym_exprs.items():
            self._partial_derivatives[element_name] = dict()
            for gen_param_name, gen_param in gen_params_dict.items():
                self._partial_derivatives[element_name][gen_param_name] = (
                    sympy.lambdify(
                        args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
                        expr=sympy.diff(element_expr, gen_param),
                        modules='numpy'
                    )
                )


    def _construct_matrix_from_blocks(self, blocks):
        # constructing one big matrix from 16 blocks
        zero_block = np.zeros((len(self._freqs), len(self._freqs)))
        return np.block([
            [blocks['NrNr'], zero_block, blocks['NrQr'], blocks['NrQi']],
            [zero_block, blocks['NiNi'], blocks['NiQr'], blocks['NiQi']],
            [blocks['QrNr'], blocks['QrNi'], blocks['QrQr'], zero_block],
            [blocks['QiNr'], blocks['QiNi'], zero_block, blocks['QiQi']]
        ])


    def compute(self, optimizing_gen_params):
        """Computes the covariance matrix at the given point.

        Computes and builds the numerical value of the covariance matrix
        at the point specified by 'optimizing_gen_params'.
        The obtained matrix will have sizes (4K+4)*(4K+4)
        and contain only numbers (not symbols).

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma_L (numpy.array): the covariance matrix evaluated at the
                4-dimensional point (specified by the 'optimizing_gen_params'
                argument of this method)
        """
        optimizing_gen_params_arrays = _construct_gen_params_arrays(
            optimizing_gen_params=optimizing_gen_params,
            freq_data_points_n=len(self._freqs)
        )

        gamma_L_blocks = dict()
        for block_name, block_function in self._elements.items():
            gamma_L_blocks[block_name] = np.diag(block_function(
                D_Ya=optimizing_gen_params_arrays['D_Ya'],
                Ef_a=optimizing_gen_params_arrays['Ef_a'],
                M_Ya=optimizing_gen_params_arrays['M_Ya'],
                X_Ya=optimizing_gen_params_arrays['X_Ya'],
                Omega_a=2.0 * np.pi * self._freqs
            ))

        gamma_L = self._construct_matrix_from_blocks(gamma_L_blocks)
        return gamma_L


    # DEPRECATED METHOD!
    # def compute_and_invert(self, optimizing_gen_params):
    #     """Computes the inverse of the covariance matrix at the given point.
    #
    #     Does exactly the same as 'compute' method but after computing
    #     the covariance matrix this method inverts the obtained matrix
    #     and returns it. See docstrings of the 'compute' method.
    #
    #     Args:
    #         optimizing_gen_params (class OptimizingGeneratorParameters):
    #             current parameters of a generator
    #             (at the current step of an optimization routine)
    #
    #     Returns:
    #         gamma_L^(-1) (numpy.array): inverted covariance matrix
    #             evaluated at the given point (specified by
    #             the 'optimizing_gen_params' argument of this method)
    #     """
    #     return sp.sparse.linalg.inv(sp.sparse.csc_matrix(
    #         self.compute(optimizing_gen_params)
    #     )).toarray()


    def compute_partial_derivatives(self, optimizing_gen_params):
        """Computes partial_derivatives of the covariance matrix.

        Each element of the covariance matrix depends on 5 quantities
        D_Ya, Ef_a, M_Ya, X_Ya (4 generator parameters) and Omega_a.
        This method constructs 4 matrices.
        The 1st matrix consists of partial derivatives at D_Ya.
        The 2nd matrix consists of partial derivatives at Ef_a.
        The 3rd matrix consists of partial derivatives at M_Ya.
        The 4th matrix consists of partial derivatives at X_Ya.
        Then it returns these 4 matrices in a dictionary.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma_L_partial_derivatives (dict): a dictionary with 4 keys:
                'D_Ya' (numpy.array): matrix of partial derivatives at D_Ya
                'Ef_a' (numpy.array): matrix of partial derivatives at Ef_a
                'M_Ya' (numpy.array): matrix of partial derivatives at M_Ya
                'X_Ya' (numpy.array): matrix of partial derivatives at X_Ya
        """
        optimizing_gen_params_arrays = _construct_gen_params_arrays(
            optimizing_gen_params=optimizing_gen_params,
            freq_data_points_n=len(self._freqs)
        )

        gamma_L_partial_derivatives = dict()
        for optimizing_gen_param_name in optimizing_gen_params_arrays.keys():
            partial_derivatives_blocks = dict()
            for block_name, partial_derivatives_functions in (
                    self._partial_derivatives.items()):
                partial_derivatives_blocks[block_name] = np.diag(
                    partial_derivatives_functions[optimizing_gen_param_name](
                        D_Ya=optimizing_gen_params_arrays['D_Ya'],
                        Ef_a=optimizing_gen_params_arrays['Ef_a'],
                        M_Ya=optimizing_gen_params_arrays['M_Ya'],
                        X_Ya=optimizing_gen_params_arrays['X_Ya'],
                        Omega_a=2.0 * np.pi * self._freqs
                    )
                )

            gamma_L_partial_derivatives[optimizing_gen_param_name] = (
                self._construct_matrix_from_blocks(partial_derivatives_blocks)
            )

        return gamma_L_partial_derivatives


    # DEPRECATED METHOD!
    # def compute_inverted_matrix_partial_derivatives(self, optimizing_gen_params):
    #     """Computes partial_derivatives of the inverted covariance matrix.
    #
    #     Each element of the inverted covariance matrix
    #     depends on 5 quantities:
    #     D_Ya, Ef_a, M_Ya, X_Ya (4 generator parameters) and Omega_a.
    #     But it is impossible to invert a big symbolic matrix
    #     (by computational reasons). That is why
    #     this method uses one math trick to compute partial_derivatives
    #     of the inverted covariance matrix at the given point
    #     without direct inverting symbolic covariance matrix.
    #
    #     Args:
    #         optimizing_gen_params (class OptimizingGeneratorParameters):
    #             current parameters of a generator
    #             (at the current step of an optimization routine)
    #
    #     Returns:
    #         inverted_gamma_L_partial_derivatives (dict): contains 4 keys:
    #             'D_Ya' (numpy.array): matrix of partial derivatives at D_Ya
    #             'Ef_a' (numpy.array): matrix of partial derivatives at Ef_a
    #             'M_Ya' (numpy.array): matrix of partial derivatives at M_Ya
    #             'X_Ya' (numpy.array): matrix of partial derivatives at X_Ya
    #     """
    #     inv_gamma_L = self.compute_and_invert(optimizing_gen_params)
    #     gamma_L_partial_derivatives = (
    #         self.compute_partial_derivatives(optimizing_gen_params)
    #     )
    #     return {
    #         'D_Ya': -inv_gamma_L @ gamma_L_partial_derivatives['D_Ya'] @ inv_gamma_L,
    #         'Ef_a': -inv_gamma_L @ gamma_L_partial_derivatives['Ef_a'] @ inv_gamma_L,
    #         'M_Ya': -inv_gamma_L @ gamma_L_partial_derivatives['M_Ya'] @ inv_gamma_L,
    #         'X_Ya': -inv_gamma_L @ gamma_L_partial_derivatives['X_Ya'] @ inv_gamma_L
    #     }



# @utils.singleton
class ObjectiveFunction:
    """Wrapper for calculations of the objective function at any point.

    Attributes:
        _R (class ResidualVector): attribute to simplify
            calculations of vector R (residual vector)
            at the current step of an optimization routine
        _gamma_L (class CovarianceMatrix): attribute to simplify
            calculations of gamma_L (covariance matrix)
            at the current step of an optimization routine
        _gen_params_prior_mean (numpy.array):
            a starting point for an optimization routine
        _inv_gamma_g (numpy.array): inverted covariance matrix
            of prior generator parameters

    Note:
        All attributes are private. Don't use them outside this class.
        Communicate with an instance of this class only via
        its public methods.
    """

    def __init__(self, freq_data,
                 gen_params_prior_mean, gen_params_prior_std_dev,
                 stage):
        """Prepares for computing the objective function at any point.

        Stores data in frequency domain, prior mean
        of generator parameters, diagonal covariance matrix
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
            stage (int): stage number (1 or 2)
                1 -- clarify generator parameters
                2 -- find the source of forced oscillations
        """
        if stage not in (1, 2):
            raise ValueError('You should specify the stage (1 or 2) '
                             'for the objective function.')
        self._stage = stage
        self._R = ResidualVector(freq_data)
        self._gamma_L = CovarianceMatrix(freq_data)
        self._gen_params_prior_mean = gen_params_prior_mean.as_array
        self._inv_gamma_g = np.diag(
            1.0 / (
                (gen_params_prior_mean.as_array *
                 gen_params_prior_std_dev.as_array)**2
            )
        )  # Why multiplied by gen_params_prior_mean.as_array (not 1.0)?


    def compute(self, optimizing_gen_params):
        """Computes value of the objective function at the given point.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            value (numpy.float64) of the objective function
                at the given point
        """
        curr_delta_params = (
            optimizing_gen_params.as_array - self._gen_params_prior_mean
        )

        computed_R = self._R.compute(optimizing_gen_params)
        computed_gamma_L = self._gamma_L.compute(optimizing_gen_params)
        computed_inv_gamma_L_dot_R = (
            np.linalg.solve(computed_gamma_L, computed_R)
        )

        return (
            curr_delta_params @ self._inv_gamma_g @ curr_delta_params
            + computed_R @ computed_inv_gamma_L_dot_R
        )


    # DEPRECATED METHOD!
    # def compute_gradient(self, optimizing_gen_params):
    #     """Computes gradient of the objective function at the given point.
    #
    #     Args:
    #         optimizing_gen_params (class OptimizingGeneratorParameters):
    #             current parameters of a generator
    #             (at the current step of an optimization routine)
    #
    #     Returns:
    #         gradient (numpy.array of 4 numbers) of the objective function
    #             at optimizing generator parameters (D_Ya, Ef_a, M_Ya, X_Ya)
    #             evaluated at the given point
    #     """
    #     curr_delta_params = (
    #         optimizing_gen_params.as_array - self._gen_params_prior_mean
    #     )
    #
    #     computed_R = self._R.compute(optimizing_gen_params)
    #     computed_gamma_L = self._gamma_L.compute(optimizing_gen_params)
    #     computed_R_partial_derivatives = (
    #         self._R.compute_partial_derivatives(optimizing_gen_params)
    #     )
    #
    #     # grad_f = grad_f1 + grad_f2 (see equation 40 in the paper)
    #     grad_f1 = (
    #         self._inv_gamma_g + np.transpose(self._inv_gamma_g)
    #     ) @ curr_delta_params
    #
    #     intermediate_grad_f2 = (
    #         np.linalg.solve(computed_gamma_L, computed_R) +
    #         np.linalg.solve(np.transpose(computed_gamma_L), computed_R)
    #     )
    #     grad_f2 = np.array([
    #         intermediate_grad_f2 @ computed_R_partial_derivatives['D_Ya'],
    #         intermediate_grad_f2 @ computed_R_partial_derivatives['Ef_a'],
    #         intermediate_grad_f2 @ computed_R_partial_derivatives['M_Ya'],
    #         intermediate_grad_f2 @ computed_R_partial_derivatives['X_Ya'],
    #     ])
    #
    #     grad_f = grad_f1 + grad_f2
    #     return grad_f


    def compute_from_array(self, optimizing_gen_params):
        """Computes value of the objective function at the given point.

        This method just calls self.compute method
        transforming the sole argument from numpy.array to an instance
        of class OptimizingGeneratorParameters. It is necessary
        to have such method because optimizers want to give an instance
        of numpy.array as an argument.

        Args:
            optimizing_gen_params (numpy.array of 4 numbers):
                current values of optimizing generator parameters
                at the current iteration of an optimization routine

        Returns:
            value (numpy.float64) of the objective function at the given point

        Note:
            Be cautious using this method! The order of parameters
            is extremely important!
        """
        print('\n### DEBUG: computing value of objective function', end=' ')
        print('at the point =', optimizing_gen_params)
        func_value = self.compute(OptimizingGeneratorParameters(
            D_Ya=optimizing_gen_params[0],
            Ef_a=optimizing_gen_params[1],
            M_Ya=optimizing_gen_params[2],
            X_Ya=optimizing_gen_params[3]
        ))
        print('### DEBUG: func_value =', func_value)
        return func_value


    # DEPRECATED METHOD!
    # def compute_gradient_from_array(self, optimizing_gen_params):
    #     """Computes gradient of the objective function at the given point.
    #
    #     This method just calls self.compute_gradient method
    #     transforming the sole argument from numpy.array to an instance
    #     of class OptimizingGeneratorParameters. It is necessary
    #     to have such method because optimizers want to give an instance
    #     of numpy.array as an argument.
    #
    #     Args:
    #         optimizing_gen_params (numpy.array of 4 numbers):
    #             current values of optimizing generator parameters
    #             (array of 4 numbers) at the current iteration
    #             of an optimization routine
    #
    #     Returns:
    #         gradient (numpy.array of 4 numbers) of the objective function
    #             at optimizing generator parameters (D_Ya, Ef_a, M_Ya, X_Ya)
    #             evaluated at the given point
    #
    #     Note:
    #         Be cautious using this method! The order of parameters
    #         is extremely important!
    #     """
    #     func_gradient = self.compute_gradient(OptimizingGeneratorParameters(
    #         D_Ya=optimizing_gen_params[0],
    #         Ef_a=optimizing_gen_params[1],
    #         M_Ya=optimizing_gen_params[2],
    #         X_Ya=optimizing_gen_params[3]
    #     ))
    #     # print('### DEBUG: gradient =', func_gradient)
    #     return func_gradient

