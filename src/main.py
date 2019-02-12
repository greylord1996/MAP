from settings import FreqData, OptimizerSettings, GeneratorParameters

freq_data = FreqData()
opt_set = OptimizerSettings()
gen_params = GeneratorParameters()

print("Freq Data: ", freq_data.get_values())
print("Optimizer Data: ", opt_set.get_values())
print("Generator Parametres: ", gen_params.get_values())
