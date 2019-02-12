import numpy as np
from settings import GeneratorParameters, PriorData

gp = GeneratorParameters()
param_dist = 1   # TO SETTINGS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
GP_vec = np.array(gp.get_list_of_values())

print('GP_vec: ', GP_vec)
print('GP_vec.size: ', GP_vec.size)
print('type; ', type(GP_vec))

rnd_vec = param_dist * np.random.rand(GP_vec.size, 1) - param_dist/2
print('rnd_vec: ', rnd_vec)
rnd_vec = rnd_vec.reshape(-1)

print('rnd_vec: ', rnd_vec)
print('rnd_vec.size: ', rnd_vec.size)
print('type_rnd: ', type(rnd_vec))

prior_mean = GP_vec + np.multiply(rnd_vec, GP_vec)
print('prior_mean: ', prior_mean)
prior_std = 1000 * np.ones(4)
print('prior_std: ', prior_std)


SNR = 45
d_coi = 0

