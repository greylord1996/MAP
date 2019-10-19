True generator parameters are represented by the following point: \
theta_g1 = 0.25 \
theta_g2 = 1.00 \
theta_g3 = 1.00 \
theta_g4 = 0.01.

We have tried to scale the objective function (multiplying factor was equal to 4, that is theta_g1 -> 4*theta_g1') along the first axis (corresponding to generator's damping).
The following results have been obtained:

-----------------

SNR = 1.0

no scaling: \
0.22797251 0.9998152  0.99090981 0.00990329 \
0.20670378 0.98730632 0.84990114 0.0117439  \
0.29872598 1.00461781 1.03981311 0.00966746 \
0.22476968 0.98618814 0.89144811 0.01089567

scaling: \
0.24867031 0.99156959 0.99337364 0.01000473 \
0.30464206 0.99135067 0.96431154 0.01027737 \
0.33263635 1.00592139 1.07899551 0.0091443  \
0.26538653 1.00233161 0.97540975 0.01043788

-----------------

SNR = 3.0

no scaling: \
0.29725007 1.00183123 1.03859207 0.00967733 \
0.26657675 1.00294831 0.95366845 0.01045154 \
0.23780116 1.0079495  0.93351754 0.01071281 \
0.15454853 0.97968565 0.86718112 0.01131271

scaling: \
0.31455192 1.00700198 1.02280521 0.0097442  \
0.19898195 1.00875111 1.08482557 0.00922983 \
0.21373413 1.01703794 1.00146505 0.01010998 \
0.20067793 0.99673131 0.95254018 0.01054965

-----------------