True generator parameters are represented by the following point: \
theta_g1 = 0.25 \
theta_g2 = 1.00 \
theta_g3 = 1.00 \
theta_g4 = 0.01.

We have tried to scale the objective function (multiplying factor was equal to 4, that is theta_g1 -> 4*theta_g1') along the first axis (corresponding to generator's damping).
The following results have been obtained:

-----------------

SNR = 1.0

no scaling : 0.31594492 0.99169888 0.9558809  0.01040966 \
scaling :    0.2104878  1.01094143 1.08854614 0.00929744

no scaling : 0.34950547 1.01576105 1.11299953 0.00918397 \
scaling :    0.34658894 0.99336018 0.93336067 0.01073942

no scaling : 0.2446188  1.00293947 1.0565174  0.00949959 \
scaling :    0.11234864 1.00638131 1.17441038 0.00866698

no scaling : 0.18959958 0.99884316 0.96658936 0.01026445 \
scaling :    0.11234864 1.00638131 1.17441038 0.00866698

-----------------

SNR = 3.0

no scaling : 0.29946774 0.99145571 0.92065478 0.01075268 \
scaling :    0.26059535 0.99926608 1.02453921 0.00979673

no scaling : 0.4874721  0.99205757 0.96135803 0.01040977 \
scaling :    0.12863362 0.98677632 0.88158884 0.01111663

no scaling : 0.22039479 0.97881665 0.87727789 0.01105401 \
scaling :    0.35757666 0.99871799 1.03137492 0.00967768

no scaling : 0.36556818 1.01935999 1.1504286  0.00883796 \
scaling :    0.29438963 1.00235495 0.98016281 0.01014578

-----------------
