import dynamic_equations_to_simulate
import data



def run_all_computations(all_params):
    ode_solver_object = dynamic_equations_to_simulate.OdeSolver(
        all_params['WhiteNoise'],
        all_params['GeneratorParameters'],
        all_params['OscillationParameters'],
        all_params['IntegrationSettings']
    )
    ode_solver_object.solve()

    ode_solver_object.simulate_time_data()
    time_data = data.TimeData(
        Vm_time_data=ode_solver_object.Vc1_abs,
        Va_time_data=ode_solver_object.Vc1_angle,
        Im_time_data=ode_solver_object.Ig_abs,
        Ia_time_data=ode_solver_object.Ig_angle
    )
    time_data.apply_white_noise(snr=45.0, d_coi=0.0)

    print('Vm_time_data =', time_data.Vm)
    print('Im_time_data =', time_data.Im)
    print('Va_time_data =', time_data.Va)
    print('Ia_time_data =', time_data.Ia)

    return ode_solver_object.get_appropr_data_to_gui()
