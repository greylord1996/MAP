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
        M_Ya (float): what is M in papers???
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



@utils.singleton
class ResidualVector:
    """Wrapper for calculations of R (residual vector).

     Based on the paper, vector R is equal to (Mr, Mi, Pr, Pi)^T.
     But the vector R is not stored. Instead of it this class stores
     only 4 functions for further computing and constructing the vector R
     at any given point on demand (see self.compute method).

    Attributes:
        _freq_data (class FreqData): data in frequency domain
        _Mr (function): for computing elements of vector Mr
        _Mi (function): for computing elements of vector Mi
        _Pr (function): for computing elements of vector Pr
        _Pi (function): for computing elements of vector Pi

    Note:
        All attributes are private. Don't change them outside this class.
        Communicate with an instance of this class only via its public methods.
    """

    def __init__(self, freq_data):
        """Prepares for computing the covariance matrix at the given point.

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

        D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = sympy.symbols(
            'D_Ya Ef_a M_Ya X_Ya Omega_a',
            real=True
        )
        args_list = [Vm, Va, Im, Ia, D_Ya, Ef_a, M_Ya, X_Ya, Omega_a]

        Mr_expr = Imr - Y11r*Vmr + Y11i*Vmi - Y12r*Var + Y12i*Vai
        Mi_expr = Imi - Y11i*Vmr - Y11r*Vmi - Y12i*Var - Y12r*Vai
        Pr_expr = Iar - Y21r*Vmr + Y21i*Vmi - Y22r*Var + Y22i*Vai
        Pi_expr = Iai - Y21i*Vmr - Y21r*Vmi - Y22i*Var - Y22r*Vai

        self._Mr = sympy.lambdify(
            args=args_list,
            expr=Mr_expr,
            modules='numexpr'
        )
        self._Mi = sympy.lambdify(
            args=args_list,
            expr=Mi_expr,
            modules='numexpr'
        )
        self._Pr = sympy.lambdify(
            args=args_list,
            expr=Pr_expr,
            modules='numexpr'
        )
        self._Pi = sympy.lambdify(
            args=args_list,
            expr=Pi_expr,
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
            R (numpy.array): residual vector (containing 4K+4 numbers)
                evaluated at the given 4-dimensional point (specified by
                the 'optimizing_gen_params' argument of this method)
        """
        D_Ya = optimizing_gen_params.D_Ya
        Ef_a = optimizing_gen_params.Ef_a
        M_Ya = optimizing_gen_params.M_Ya
        X_Ya = optimizing_gen_params.X_Ya

        freq_data_len = len(self._freq_data.freqs)
        Mr = np.zeros(freq_data_len)
        Mi = np.zeros(freq_data_len)
        Pr = np.zeros(freq_data_len)
        Pi = np.zeros(freq_data_len)

        for i in range(freq_data_len):
            Mr[i] = self._Mr(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                D_Ya, Ef_a, M_Ya, X_Ya,
                self._freq_data.freqs[i]
            )
            Mi[i] = self._Mi(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                D_Ya, Ef_a, M_Ya, X_Ya,
                self._freq_data.freqs[i]
            )
            Pr[i] = self._Pr(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                D_Ya, Ef_a, M_Ya, X_Ya,
                self._freq_data.freqs[i]
            )
            Pi[i] = self._Pi(
                self._freq_data.Vm[i], self._freq_data.Va[i],
                self._freq_data.Im[i], self._freq_data.Ia[i],
                D_Ya, Ef_a, M_Ya, X_Ya,
                self._freq_data.freqs[i]
            )

        # Build and return R (residual vector)
        return np.concatenate([Mr, Mi, Pr, Pi])



@utils.singleton
class CovarianceMatrix:
    """Wrapper for calculations of covariance matrix at any point.

    Attributes:
        _freqs (np.array): frequencies in frequency domain
        _elements (dict): contains 12 keys:
            'NrNr', 'NrQr', 'NrQi', 'NiNi', 'NiQr', 'NiQi',
            'QrQr', 'QiQi', 'QrNr', 'QrNi', 'QiNr', 'QiNi'.
            every key matches to function for computing
            corresponding element of the gamma_L matrix
        _partial_derivatives (dict of dicts): contains 12 keys:
            'NrNr', 'NrQr', 'NrQi', 'NiNi', 'NiQr', 'NiQi',
            'QrQr', 'QiQi', 'QrNr', 'QrNi', 'QiNr', 'QiNi'.
            every key matches to dictionary holding 4 functions:
            'D_Ya': (function) for computing partial derivative at D_Ya
            'Ef_a': (function) for computing partial derivative at Ef_a
            'M_Ya': (function) for computing partial derivative at M_Ya
            'X_Ya': (function) for computing partial derivative at X_Ya

    Note:
        All attributes are private. Don't change them outside this class.
        Communicate with an instance of this class only via its public methods.
    """

    def __init__(self, freq_data):
        """Prepares the covariance matrix for computing at any point.

        Stores data in frequency domain, 12 compiled functions
        (see sympy.lambdify) for further computing and constructing
        the gamma_L matrix at any 4-dimensional point (the number of parameters
        which we optimize is equal to 4) and 48 compiled functions
        for computing and constructing 4 matrices (each of the 4 matrices
        contains partial derivatives at D_Ya, Ef_a, M_Ya, X_Ya respectively).

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
                modules='numexpr'
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
                partial_derivative_expr = sympy.diff(element_expr, gen_param)

                # It seems that numexpr module doesn't support sign function
                # (see https://github.com/pydata/numexpr/issues/87).
                # That is why we replace sign(x) with |x| / x
                # (we have checked that x is always a nonzero real number
                # in our code). Moreover, x is always greater than 0 because
                # it always looks like x = Ef_a**2 - 1.75516512378075*Ef_a + 1,
                # where Ef_a is a real number.
                partial_derivative_expr = partial_derivative_expr.replace(
                    sympy.sign, lambda x: sympy.Abs(x) / x
                )

                self._partial_derivatives[element_name][gen_param_name] = (
                    sympy.lambdify(
                        args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
                        expr=partial_derivative_expr,
                        modules='numexpr'
                    )
                )


    def _construct_matrix_from_blocks(self, blocks):
        # Constructing one big matrix from 16 blocks
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
        at the point specified by 'optimizing_gen_params'. The obtained matrix
        will have sizes (4K+4)*(4K+4) and contain only numbers (not symbols).

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma_L (numpy.ndarray): the covariance matrix evaluated at the
            given 4-dimensional point (specified by the 'optimizing_gen_params'
            argument of this method)
        """
        D_Ya = optimizing_gen_params.D_Ya
        Ef_a = optimizing_gen_params.Ef_a
        M_Ya = optimizing_gen_params.M_Ya
        X_Ya = optimizing_gen_params.X_Ya

        gamma_L_blocks = dict()
        for block_name, element_function in self._elements.items():
            gamma_L_blocks[block_name] = np.diag([
                element_function(D_Ya, Ef_a, M_Ya, X_Ya, 2.0 * np.pi * freq)
                for freq in self._freqs
            ])

        gamma_L = self._construct_matrix_from_blocks(gamma_L_blocks)
        return gamma_L


    def compute_and_invert(self, optimizing_gen_params):
        """Computes the inverse of the covariance matrix at the given point.

        Does exactly the same as 'compute' method but after computing
        the covariance matrix this method inverts the obtained matrix
        and returns it. See docstrings of the 'compute' method.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma_L^(-1) (numpy.ndarray): inverted covariance matrix
                evaluated at the given point (specified by
                the 'optimizing_gen_params' argument of this method)
        """
        return sp.sparse.linalg.inv(sp.sparse.csc_matrix(
            self.compute(optimizing_gen_params)
        )).toarray()


    def compute_gradient(self, optimizing_gen_params):
        """Computes gradient of the covariance matrix at the given point.

        Each element of the covariance matrix depends on 4 quantities
        (4 generator parameters). This method constructs 4 matrices.
        The 1st matrix consists of partial derivatives at the D_Ya.
        The 2nd matrix consists of partial derivatives at the Ef_a.
        The 3rd matrix consists of partial derivatives at the M_Ya.
        The 4th matrix consists of partial derivatives at the X_Ya.
        Then it returns the 4 matrices in a dictionary.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gamma_L_gradient (dict): a dictionary with 4 keys:
                'D_Ya' (numpy.ndarray): matrix of partial derivatives at D_Ya
                'Ef_a' (numpy.ndarray): matrix of partial derivatives at Ef_a
                'M_Ya' (numpy.ndarray): matrix of partial derivatives at M_Ya
                'X_Ya' (numpy.ndarray): matrix of partial derivatives at X_Ya
        """
        optimizing_gen_params_dict = {
            'D_Ya': optimizing_gen_params.D_Ya,
            'Ef_a': optimizing_gen_params.Ef_a,
            'M_Ya': optimizing_gen_params.M_Ya,
            'X_Ya': optimizing_gen_params.X_Ya
        }

        gamma_L_gradient = dict()
        for optimizing_gen_param_name in optimizing_gen_params_dict.keys():
            gradient_blocks = dict()
            for block_name, partial_derivatives_functions in (
                    self._partial_derivatives.items()):
                gradient_blocks[block_name] = np.diag([
                    partial_derivatives_functions[optimizing_gen_param_name](
                        optimizing_gen_params_dict['D_Ya'],
                        optimizing_gen_params_dict['Ef_a'],
                        optimizing_gen_params_dict['M_Ya'],
                        optimizing_gen_params_dict['X_Ya'],
                        2.0 * np.pi * freq
                    ) for freq in self._freqs
                ])

            gamma_L_gradient[optimizing_gen_param_name] = (
                self._construct_matrix_from_blocks(gradient_blocks)
            )

        return gamma_L_gradient


    def compute_inverted_matrix_gradient(self, optimizing_gen_params):
        """Computes gradient of the inverted covariance matrix.

        Each element of the inverted covariance matrix depends on 4 quantities
        (4 generator parameters). But it is impossible to invert
        a big symbolic matrix (by computational reasons). That is why
        this method uses one math trick to compute gradient
        of the inverted covariance matrix at the given point
        without direct inverting symbolic covariance matrix.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            inverted_gamma_L_gradient (dict): a dictionary with 4 keys:
                'D_Ya' (numpy.ndarray): matrix of partial derivatives at D_Ya
                'Ef_a' (numpy.ndarray): matrix of partial derivatives at Ef_a
                'M_Ya' (numpy.ndarray): matrix of partial derivatives at M_Ya
                'X_Ya' (numpy.ndarray): matrix of partial derivatives at X_Ya
        """
        inv_gamma_L = self.compute_and_invert(optimizing_gen_params)
        gamma_L_gradient = self.compute_gradient(optimizing_gen_params)
        return {
            'D_Ya': -inv_gamma_L @ gamma_L_gradient['D_Ya'] @ inv_gamma_L,
            'Ef_a': -inv_gamma_L @ gamma_L_gradient['Ef_a'] @ inv_gamma_L,
            'M_Ya': -inv_gamma_L @ gamma_L_gradient['M_Ya'] @ inv_gamma_L,
            'X_Ya': -inv_gamma_L @ gamma_L_gradient['X_Ya'] @ inv_gamma_L
        }



@utils.singleton
class ObjectiveFunction:
    """Wrapper for calculations of the objective function at any point.

    Attributes:
        _R (class ResidualVector): attribute to simplify
            calculations of vector R (residual vector)
            at the current step of an optimization routine
        _gamma_L (class CovarianceMatrix): attribute to simplify
            calculations of gamma_L (covariance matrix)
            at the current step of an optimization routine
        _gen_params_prior_means (numpy.array):
            a starting point for an optimization routine
        _reversed_gamma_g (numpy.ndarray): diagonal matrix containing
            standard deviations of prior uncertain generator parameters

    Note:
        All attributes are private. Don't change them outside this class.
        Communicate with an instance of this class only via its public methods.
    """

    def __init__(self, freq_data,
                 gen_params_prior_means, gen_params_prior_std_devs):
        """Prepares for computing the objective function at any point.

        Stores data in frequency domain, prior means of generator parameters,
        diagonal covariance matrix of its parameters. Prepares the vector R
        (residual vector) and gamma_L (covariance matrix) for computing
        at any point.

        Args:
            freq_data (class FreqData): data in frequency domain
            gen_params_prior_means (class OptimizingGeneratorParameters):
                prior means of generator's parameters
                (we are uncertain in their values)
            gen_params_prior_std_devs (class OptimizingGeneratorParameters):
                standard deviations of generator's parameters
                (how much we are uncertain in their values)
        """
        self._R = ResidualVector(freq_data)
        self._gamma_L = CovarianceMatrix(freq_data)
        self._gen_params_prior_means = gen_params_prior_means.as_array
        self._reversed_gamma_g = np.diag(gen_params_prior_std_devs.as_array)


    def compute(self, optimizing_gen_params):
        """Computes value of the objective function at the given point.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            value (numpy.float64) of the objective function at the given point
        """
        curr_delta_params = (
            optimizing_gen_params.as_array - self._gen_params_prior_means
        )

        computed_R = self._R.compute(optimizing_gen_params)
        computed_inverted_gamma_L = (
            self._gamma_L.compute_and_invert(optimizing_gen_params)
        )

        # print('*********** COMPUTING GAMMA_L')
        # computed_gamma_L = self._gamma_L.compute(uncertain_gen_params)
        # print('*** COND NUMBER OF GAMMA_L =', np.linalg.cond(computed_gamma_L))
        # print('*********** *****************')

        return (
            curr_delta_params @ self._reversed_gamma_g @ curr_delta_params
            + computed_R @ computed_inverted_gamma_L @ computed_R
        )


    def compute_gradient(self, optimizing_gen_params):
        """Computes gradient of the objective function at the given point.

        Args:
            optimizing_gen_params (class OptimizingGeneratorParameters):
                current parameters of a generator
                (at the current step of an optimization routine)

        Returns:
            gradient (numpy.array) of the objective function at the given point
        """
        pass


    def compute_by_array(self, optimizing_gen_params):
        """Computes value of the objective function at the given point.

        This method just calls self.compute method
        transforming the sole argument from numpy.array to an instance
        of class OptimizingGeneratorParameters. It is necessary
        to have such method because optimizers want to give an instance
        of numpy.array as an argument.

        Args:
            optimizing_gen_params (numpy.array):
                current values of optimizing generator parameters
                (at the current iteration of an optimization routine)

        Returns:
            value (numpy.float64) of the objective function at the given point

        Note:
            Be cautious using this method! The order of parameters
            is extremely important!
        """
        # print('### DEBUG: optimizing... curr_point =', optimizing_gen_params)
        return self.compute(OptimizingGeneratorParameters(
            D_Ya=optimizing_gen_params[0],
            Ef_a=optimizing_gen_params[1],
            M_Ya=optimizing_gen_params[2],
            X_Ya=optimizing_gen_params[3]
        ))


    def compute_gradient_by_array(self, optimizing_gen_params):
        """Computes gradient of the objective function at the given point.

        This method just calls self.compute_gradient method
        transforming the sole argument from numpy.array to an instance
        of class OptimizingGeneratorParameters. It is necessary
        to have such method because optimizers want to give an instance
        of numpy.array as an argument.

        Args:
            optimizing_gen_params (numpy.array):
                current values of optimizing generator parameters
                (at the current iteration of an optimization routine)

        Returns:
            gradient (numpy.array) of the objective function at the given point

        Note:
            Be cautious using this method! The order of parameters
            is extremely important!
        """
        pass

