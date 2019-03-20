from create_admittance_matrix import AdmittanceMatrix
from sympy import *

class ResidualVector:

    def __init__(self):
        pass


class CovarianceMatrix:

    def __init__(self, std_eps_Vm, std_eps_Va, std_eps_Im, std_eps_Ia):
        self.std_eps_Vm = std_eps_Vm
        self.std_eps_Va = std_eps_Va
        self.std_eps_Im = std_eps_Im
        self.std_eps_Ia = std_eps_Ia

        self.admittance_matrix = AdmittanceMatrix().Ys

        self.Y11 = self.admittance_matrix[0, 0]
        self.Y12 = self.admittance_matrix[0, 1]
        self.Y21 = self.admittance_matrix[1, 0]
        self.Y22 = self.admittance_matrix[1, 1]

        self.Y11r = re(self.Y11)
        self.Y11i = im(self.Y11)
        self.Y12r = re(self.Y12)
        self.Y12i = im(self.Y12)
        self.Y21r = re(self.Y21)
        self.Y21i = im(self.Y21)
        self.Y22r = re(self.Y22)
        self.Y22i = im(self.Y22)

        self.gamma_NrNr = None
        self.gamma_NrQr = None

    def _init_gamma_NrNr(self, omega):
        self.gamma_NrNr = (
            self.std_eps_Im**2
            + self.std_eps_Vm**2 * (self.Y11r**2 + self.Y11i**2)
            + self.std_eps_Va**2 * (self.Y12r**2 + self.Y12i**2)
        )




