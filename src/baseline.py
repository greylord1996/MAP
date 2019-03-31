import dynamic_equations_to_simulate
import data
import objective_function



def run_all_computations(all_params):
    ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
        white_noise=all_params['WhiteNoise'],
        gen_param=all_params['GeneratorParameters'],
        osc_param=all_params['OscillationParameters'],
        integr_param=all_params['IntegrationSettings']
    )
    # ode_solver_object.solve()

    ode_solver_object.simulate_time_data()
    time_data = data.TimeData(
        Vm_time_data=ode_solver_object.Vc1_abs,
        Va_time_data=ode_solver_object.Vc1_angle,
        Im_time_data=ode_solver_object.Ig_abs,
        Ia_time_data=ode_solver_object.Ig_angle,
        dt=ode_solver_object.dt
    )
    time_data.apply_white_noise(snr=45.0, d_coi=0.0)

    freq_data = data.FreqData(time_data)

    prior_gen_params = objective_function.UncertainGeneratorParameters(
        Ef_a=1.0, D_Ya=2.0, X_Ya=3.0, M_Ya=4.0,
        std_dev_Ef_a=0.1, std_dev_D_Ya=0.2, std_dev_X_Ya=0.3, std_dev_M_Ya=0.4
    )

    # f denotes the objective function
    f = objective_function.ObjectiveFunction(
        freq_data=freq_data,
        prior_gen_params=prior_gen_params
    )

    # Here we should minimize the objective function
    # opt_res = sp.optimize.minimize(
    #     fun=f.compute_by_array,
    #     x0=prior_gen_params.get_as_array(),
    #     method='BFGS',
    #     # tol=...?
    #     options={
    #         # 'maxiter': 1,
    #         'disp': True
    #     }
    # )

    # It is not clear now what should be returned
    return ode_solver_object.get_appropr_data_to_gui()
