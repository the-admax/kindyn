import typing as ty
from scipy.integrate import RK45, DenseOutput
import numpy as np


class IntegrationError(Exception):
    pass



def my_odeint(f, y0, t0, t1,
              piece_fn: ty.Optional[ty.Callable[[DenseOutput], ty.Optional[np.ndarray]]]=None,
              *,
              max_step=np.inf,
              on_progress=None):
    """
    Explicit 4th order Runge-Kutta integrator with ability to do piecewise approximation used in collision handling.

    :param f:   Derivative function of signature `(t, y)`
    :param y0:  Initial conditions (state)
    :param t0:  Starting point of time to integrate the path from
    :param t1:  Ending point of time (incl.)
    :param piece_fn: a function that supplies new initial state for integrator. It receives :class:`DenseOutput`
        instance from integrator to be able to check the conditions. If it returns False, the integration gets
        interrupted, if `(t, y)`, they are taken as new initial state, otherwise (None) - ignored.
    :param on_progress: A function that is called each time the time is advanced. It receives the current time
        of simulation
    """

    y0 = np.asarray(y0)

    ts = []
    ys = []
    derivs = []
    should_run = True
    ignore_next = False     # This flag is set when peice_fn triggered at the either edge of the step.
                            # it forces integration step to prevent multiple application of piece_fn.

    ts.append(t0)
    ys.append(y0)
    derivs.append(f(t0, y0))

    while should_run:
        solver = RK45(f, t0, y0, t1, max_step=max_step)

        # while (t < t1) or (reverse_dir and t1 < t):
        while solver.status == 'running':
            failure_reason = solver.step()

            if solver.status == 'failed':
                raise IntegrationError(failure_reason)

            if on_progress is not None:
                on_progress(solver.t)

            if piece_fn is not None and not ignore_next:
                res = piece_fn(solver.dense_output())
                if res is False:
                    should_run = False
                    break
                if res is not None:
                    ignore_next = not (t0 < res[0] < t1)
                    t0, y0 = res    # restart from new point
                    del solver      # Free some memory
                    break  # By jumping to outer loop

            ignore_next = False

            ts.append(solver.t)
            ys.append(solver.y)
            derivs.append(f(solver.t, solver.y))

            if solver.status == 'finished':
                should_run = False

    return np.array(ts), np.array(ys), np.array(derivs)