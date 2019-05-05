from sympy import *
import numexpr


D_Ya, Ef_a, M_Ya, X_Ya, Omega_a = symbols(
    'D_Ya Ef_a M_Ya X_Ya Omega_a',
    real=True
)


NrNr_Ef_a = (

7.639e-13*(
    - 0.229*Ef_a**2*
    re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1))) / (M_Ya*X_Ya**2)
    + (1 - 0.877*Ef_a)*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))**2

 + 1.200e-8*(0.4207*Ef_a**2*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) - 0.479*Ef_a*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))**2

 + 7.639e-13*(-0.229*Ef_a**2*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) - (1 - 0.877*Ef_a)*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))**2

 + 1.200e-8*(0.420*Ef_a**2*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) + 0.479*Ef_a*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))**2

 + 6.291e-5

)


# gradient_NrNr_Ef_a = diff(NrNr_Ef_a, Ef_a)

gradient_NrNr_Ef_a = (

7.639e-13*(-0.229*Ef_a**2*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) + (1 - 0.877*Ef_a)*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))
* (-0.458*Ef_a**2*((0.8775 - Ef_a)*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*(Ef_a**2 - 1.755*Ef_a + 1)**(3/2))) - 0.877*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))**2*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya))/(M_Ya*X_Ya**2) - 0.916*Ef_a*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) - (1 - 0.877*Ef_a)*(2*Ef_a - 1.755)*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)*sign(Ef_a**2 - 1.755*Ef_a + 1)/(X_Ya*Abs(Ef_a**2 - 1.755*Ef_a + 1)**(3/2)) - 1.754*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))

+ 1.2e-8*(0.4207*Ef_a**2*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) - 0.479*Ef_a*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))
* (0.8414*Ef_a**2*((0.8775 - Ef_a)*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*(Ef_a**2 - 1.755*Ef_a + 1)**(3/2))) - 0.877*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))**2*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya))/(M_Ya*X_Ya**2) + 0.479*Ef_a*(2*Ef_a - 1.755)*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)*sign(Ef_a**2 - 1.755*Ef_a + 1)/(X_Ya*Abs(Ef_a**2 - 1.755*Ef_a + 1)**(3/2)) + 1.6828*Ef_a*re(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) - 0.958*cos(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))

+ 7.639e-13*(-0.229*Ef_a**2*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) - (1 - 0.877*Ef_a)*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))
* (-0.458*Ef_a**2*((0.8775 - Ef_a)*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*(Ef_a**2 - 1.755*Ef_a + 1)**(3/2))) - 0.877*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))**2*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya))/(M_Ya*X_Ya**2) - 0.916*Ef_a*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) + (1 - 0.877*Ef_a)*(2*Ef_a - 1.755)*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)*sign(Ef_a**2 - 1.755*Ef_a + 1)/(X_Ya*Abs(Ef_a**2 - 1.755*Ef_a + 1)**(3/2)) + 1.754*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))

+ 1.2e-8*(0.42*Ef_a**2*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) + 0.479*Ef_a*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))
* (0.84*Ef_a**2*((0.8775 - Ef_a)*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*(Ef_a**2 - 1.755*Ef_a + 1)**(3/2))) - 0.877*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))**2*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya))/(M_Ya*X_Ya**2) - 0.479*Ef_a*(2*Ef_a - 1.755)*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)*sign(Ef_a**2 - 1.755*Ef_a + 1)/(X_Ya*Abs(Ef_a**2 - 1.755*Ef_a + 1)**(3/2)) + 1.68*Ef_a*im(1/((0.877*Ef_a/(M_Ya*X_Ya) + I*Omega_a*(D_Ya/M_Ya + I*Omega_a))*sqrt(Ef_a**2 - 1.755*Ef_a + 1)))/(M_Ya*X_Ya**2) + 0.958*sin(atan2(0, Ef_a**2 - 1.755*Ef_a + 1)/2)/(X_Ya*sqrt(Abs(Ef_a**2 - 1.755*Ef_a + 1))))

)


compiled_gradient_NrNr_Ef_a = lambdify(
    args=[D_Ya, Ef_a, M_Ya, X_Ya, Omega_a],
    expr=diff(gradient_NrNr_Ef_a, Ef_a),
    modules='numexpr'
)

