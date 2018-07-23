from functools import partial
# from progressbar.progressbar import ProgressBar
# import progressbar.widgets as pgbar_widgets

from kindyn.params import *
from kindyn import *

if __name__ == '__main__':
    ##
    ## Setup the world
    ##
    inside_sphere = mk_sphere(a, b)
    v_circ = mk_v_circ(G, a, b, M)

    # Parametrise functions of potentials
    halo_potential_f = mk_halo_potential(v_h, r_c, q)
    gas_potential_f = mk_miyamoto_nagai_pot(G, a, b, M)

    halo_gravity = getForce(halo_potential_f)
    gas_gravity = getForce(gas_potential_f)

    rhoM_N = mk_gas_density(G, a, b, M)

    gas_friction = mk_chandrasekhar_acc(G, a, b, logLambda, rhoM_N, sigma, v_circ)
    halo_friction = mk_chandrasekhar_acc(G, r_c, r_c, logLambda, mk_halo_density(G, v_h, q, r_c), sigma)

    # mass_accretion = mk_mass_accretion(m_dot_f)

    mdot = mk_mdot(G, v_s, m_tot, rhoM_N)
    mass_accretion = mk_mass_accretion(mdot)    # Accretion - acceleration due to mass loss

    mass_loss = mk_mass_loss(M, ra0_M31, m32_mass, r32_init, sigma_m32, W=r32_init/50,
                             dir=np.sign(t_final - t_init))

    interactions = [
        halo_gravity,
        gas_gravity,
        halo_friction,
        gas_friction,
        # mass_loss,
        # mass_accretion,
    ]

    total_accel = lambda t, s: np.sum(np.array([f(*s) for f in interactions]),
                                      axis=0) + np.array([s[3], s[4], s[5], 0, 0, 0, 0])

    # Initialize state as (r0x, r0y, r0z, v0x, v0y, v0z, m0) == [x,y,z,u,v,w,m]
    s_init = np.concatenate((r_init, v_init, [m32_mass]))  # начальные данные

    # Интегрирование по времени с одинаковым интервалом
    print("Initial state [", t_init, "] = ", s_init)
    # pgbar = ProgressBar(maxval=t_final - t_init, widgets=[
    #     pgbar_widgets.Percentage(),
    #     pgbar_widgets.Bar(),
    #     pgbar_widgets.AdaptiveETA()
    # ])
    # pgbar.start()

    from kindyn.my_ode import my_odeint
    from kindyn.my_ode2 import my_odeint as my_odeint2, IntegrationError

    ts0, path0, derivs0 = my_odeint(total_accel, s_init, t_init, t_final, n_steps,  # , hmax=2*(t_final-t_init)/n_steps, atol=1e-4, )
                                 # on_progress=lambda t: pgbar.update(t - t_init)
                                    )
    ts, path, derivs = my_odeint2(total_accel, s_init, t_init, t_final,
                                     partial(pericenter_massloss, mass_loss),
                                     max_step=np.abs(t_final - t_init)/n_steps,
                                  #   on_progress=lambda t: pgbar.update(t - t_init)
                                  )

    n = min(path.shape[0], ts.shape[0])
    print("Final state [", ts[n - 1], "] = ", path[-1, :])

    print("N pts = {0}/{1} [{2:.2f}%] (last t={3})".format(path.shape[0], ts.shape[0],
                                                           100 * path.shape[0] / ts.shape[0],
                                                           ts[min(n, len(ts) - 1)]))

    plt.switch_backend('Qt5Agg')
    plt.interactive(True)
    fig = plt.figure(figsize=(10., 10.))
    kd = KinDyn(fig)
    fig.add_subplot()
    ax3d = kd.plot_results((2, 2, 1), path0, label="Old method (ode)", color='black')
    kd.plot_results(ax3d, path, label="New method (RK45)", color='red')

    ax_mass = fig.add_subplot(2, 2, 2)
    ax_mass.scatter(LA.norm(path[:,0:3] * 1e-3, axis=1), path[:,6],
                    label=r"dm / z", alpha=0.2)
    ax_mass.set_xlabel(r'R, kpc')
    ax_mass.set_ylabel(r'$\Delta M, 10^6\ M_\odot$')

    ax_r = fig.add_subplot(2, 2, 3)
    ax_r.plot(ts, LA.norm(path[:,0:3] * 1e-3, axis=1), label=r"R(t)")
    ax_r.set_xlabel(r't, Myr')
    ax_r.set_ylabel(r'R, kpc')
    ax_r.tick_params('y', colors='b')

    ax_m = ax_r.twinx()
    ax_m.plot(ts, path[:, 6], 'r')
    ax_m.set_ylabel(r'M, 10^6\ M_\odot', color='r')
    ax_m.tick_params('y', colors='r')

    ax_Lz = fig.add_subplot(2, 2, 4)
    R, V, m = path[:n,0:3], path[:n,3:6], path[:n,6]
    Lz = m * np.abs(np.cross(R[:,0:2], V[:,0:2]))
    ax_Lz.plot(ts, Lz, label=r"|Lz(t)|")
    ax_Lz.set_xlabel(r't, Myr')
    ax_Lz.set_ylabel(r'Lz, ?')


    fig.legend()
    plt.show(block=True)
    # input("Press enter to exit")

    # print('-------')

    # for n in [k for k in locals() if not k.startswith('_') and k not in hidden_locals]:
    #     print("{}: {!r}".format(n,locals().get(n) ))
    # print('=======')
