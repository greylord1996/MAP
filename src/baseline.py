import dynamic_equations_to_simulate
import data



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

    return ode_solver_object.get_appropr_data_to_gui()
