import numpy as np
from scipy.integrate import ode

def my_odeint(f, y0, t0, t1, n_steps,*, on_progress=None):
    """
    ODE integrator compatible with odeint, but uses ode underneath
    """

    y0 = np.asarray(y0)

    solver = ode(f)
    solver.set_integrator("dopri5", nsteps=2000) 

    # t_series = np.linspace(t0, t1, n_steps)
    dt = (t1 - t0) / n_steps

    solver.set_initial_value(y0, t0)

    t_series = [t0]
    y_result = [y0]
    derivs = [f(t0, y0)]

    i = 1
    t = t0

    while solver.successful():
        solver.integrate(t)

        if on_progress is not None:
            on_progress(solver.t)

        t_series.append( solver.t)
        y_result.append( solver.y)
        derivs.append(f(solver.t, solver.y))

        i += 1
        if i < n_steps:
            t += dt

    return np.array(t_series), np.array(y_result), np.array(derivs)

