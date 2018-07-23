#from numba import jit
from numpy import linalg as LA
import numpy as np
import sympy as sp
import typing as ty
from itertools import count
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import DenseOutput
from scipy.optimize import brentq

# import astropy.units as u

__author__ = 'admax'

def jit(fn, *args, **kwargs):
    return fn

def getForce(pot_f):
    """
    Берёт на вход функцию потенциала ``f(x, y, z, u, v, w, m)``
        где (x,y,z) -- положение в пр-ве, (u,v,w) -- лин. скорость, m - масса
    Возвращает производную функции потенциала.
    """
    x, y, z, u, v, w, m = syms = sp.symbols('x, y, z, u, v, w, m')  # символы для sp.
    pot_expr = pot_f(x, y, z, u, v, w, m)

    df_dx = sp.lambdify(syms, -sp.diff(pot_expr, x))
    df_dy = sp.lambdify(syms, -sp.diff(pot_expr, y))
    df_dz = sp.lambdify(syms, -sp.diff(pot_expr, z))

    print(-sp.diff(pot_expr, x))

    def deriv_f(x, y, z, u, v, w, m):  # функция, выч. производные для конкр. точки.
        return np.array([0, 0, 0,
                         df_dx(x, y, z, u, v, w, m),
                         df_dy(x, y, z, u, v, w, m),
                         df_dz(x, y, z, u, v, w, m),
                         0], dtype=np.float)

    # "меняет скорость, но не положение напрямую"

    return deriv_f


def mk_gas_density(G, a, b, M):
    def gas_density(x, y, z):
        R2 = x * x + y * y + z * z
        rho = M * b * b * (a * R2 + (a + 3 * (z * z + b * b) ** .5) * (a + (z * z + b * b) ** .5) ** 2)
        rho /= 4 * np.pi * ((R2 + (a + (z * z + b * b) ** .5) ** 2) ** 2.5) * ((z * z + b * b) ** 1.5)
        #         print(x,y,z, rho)
        return rho

    return gas_density


def mk_miyamoto_nagai_pot(G, a, b, M):
    """ потенциал м-нагаи
    """

    def miyamoto_nagai_pot(x, y, z, u, v, w, m):
        return -M * G / sp.sqrt(x * x + y * y + (a + sp.sqrt(z * z + b * b)) ** 2)

    return miyamoto_nagai_pot


def mk_halo_density(G, v_h, q, r_c):
    def halo_density(x, y, z):
        rho = 2 * q * q * v_h * v_h * (
                    r_c * r_c * q * q * (2 * q * q + 1) - 2 * q * q * q * q * (x * x + y * y) + q * q * (
                        x * x + y * y + 2 * z * z) - 3 * z * z)
        rho /= 4 * np.pi * G * (r_c * r_c * q * q + q * q * (x * x + y * y) + z * z) ** 3
        return rho

    return halo_density


def mk_halo_potential(v_h, r_c, q=1.0):
    """ Logarithmic dark matter halo potential
    Ref:
    """

    def halo_potential(x, y, z, u, v, w, m):
        return v_h * v_h * sp.log(x * x + y * y + z * z / (q * q) + r_c * r_c)

    return halo_potential


# -------------------------------------Дополнения------------------------
"функция v circ (может стоит засунуть в квадрат, но пусть пока будет как есть)"


def mk_v_circ(G, a, b, M):
    """
    :param a -- радиус диска
    :param b -- полутолщина диска (=толщина/2)
    """

    def v_circ(x, y, z):
        """#FLAT#может z-координата и не нужна (https://lavinia.as.arizona.edu/~gbesla/ASTR_400B_files/BonusAssign9.pdf)
        Возвращает вектор скорости движения диска в точке x,y,z
        """
        R2 = x * x + y * y
        r = R2 ** 0.5

        vc = ((M * G * R2) / ((R2 + (a + b) ** 2) ** 1.5)) ** 0.5
        # Вектор скорости кольца направлен против часовой стрелки (перпендикулярно влево, если смотреть
        # с вершины оси Z) в плоскости диска от точки (x,y,z). Чтобы завернуть диск по часовой стрелке (вправо),
        # умножаем на `-vc`, а не на `vc`.
        return vc * np.array([-y / r, x / r, 0])

    return v_circ


def mk_sphere(a, b):
    """ Сделать ф-цию, проверяющую вхождение в эллипсоид симметричный по осям Х-Y
    """

    @np.vectorize
    def inside_sphere(x, y, z):

        #         k=0.1
        #         r = x**2/a**2 + y**2 /a**2 + z**2/b**2
        #         c= 0.5+0.5*np.tanh((1-r)/k))
        if (x ** 2 / a ** 2 + y ** 2 / a ** 2 + z ** 2 / b ** 2 <= 1):  # or True:
            return 1.0
        else:
            return 0.0

    return inside_sphere


#     return c


def mk_chandrasekhar_acc(G, a, b, logLambda, rho_f, sigma, v_circ=None, slope=1e-4):
    """ Фукнкция динамечиского трения
    :param a -- radius in XY plane
    :param b -- radius at Z axis
    """
    k1 = 1 / (np.sqrt(2) * sigma)
    k2 = -4 * np.pi * G * G * logLambda
    a2 = 1 / a * a
    b2 = 1 / b * b

    @np.vectorize
    def inside_sphere(x, y, z):
        #         r = x**2/a**2 + y**2 /a**2 + z**2/b**2
        #         c= 0.5+0.5*np.tanh((1-r)/slope))
        return ((x ** 2 + y ** 2) * a2 + z ** 2 * b2 <= 1) or 1

    from math import erf

    def acceleration(x, y, z, u, v, w, m):
        if inside_sphere(x, y, z):
            v_norm = np.sqrt(u * u + v * v + w * w)
            r_norm = np.sqrt(x * x + y * y + z * z)

            c0 = v_norm * k1
            V = np.array([u, v, w])

            if v_circ is not None:
                # vv -- скорость скопления в момент врезания во вращающийся диск
                vv = V - v_circ(x, y, z)
                vv_norm = LA.norm(vv)
            else:
                vv_norm = v_norm

            #             c = (erf(c0) - 2.*c0*np.exp(-.5*c0*c0)/np.sqrt(np.pi))*np.clip(v_norm-v_circ(x, y, z), 1, 1e6)**-3

            c = (erf(c0) - 2. * c0 * np.exp(-.5 * c0 * c0) / np.sqrt(np.pi)) * vv_norm ** -3

            a = k2 * m * V * rho_f(x, y, z) * c
            #         m_dot = np.zeros(1)
            #         veloc = np.zeros(3)
            return np.array([0, 0, 0, a[0], a[1], a[2], 0])
        else:
            return np.zeros(7)

    return acceleration


def mk_mdot(G, v_s, m_tot, rho_f):
    def mdot(x, y, z, u, v, w, m):
        if m_tot > m:
            #         return 4*np.pi*(m_tot-m)*rho(x,y,z)*G*G*m*m*(v_s*v_s+u*u+v*v+w*w)**-1.5 # аккреция
            return 4 * np.pi * (m_tot - m) * rho_f(x, y, z) * G * G * m / (
                        v_s * v_s + u * u + v * v + w * w) ** 1.5  # аккреция
        #             return 10**-15*m**-2*rho_f(x,y,z)*(v_s*v_s + u*u + v*v + w*w)**-3 # аккреция по тутукову
        return 0

    return mdot


def mk_mass_accretion(mdot_f):
    """ Возвращает функцию ускорения и изменения массы, связанных с акрецией.
    Вроде как нормальная ф-ция (там так же -> https://github.com/astroandes/BlackHole/blob/master/src/test.ipynb)
    """

    def mass_accretion(x, y, z, u, v, w, m):
        if inside_sphere(x, y, z):
            dm = - 0.01 * mdot_f(x, y, z, u, v, w, m)
            # -dm*v/m -- это аккреция, и её вклад в ускорение движущегося тела

            #     a_x = (Hx(x,y,z)+(m_tot-m)*Gx(x,y,z)-dm*u)/m
            #     a_y = (Hy(x,y,z)+(m_tot-m)*Gy(x,y,z)-dm*v)/m
            #     a_z = (Hz(x,y,z)+(m_tot-m)*Gz(x,y,z)-dm*w)/m
            return np.array([0, 0, 0, -dm * u / m, -dm * v / m, -dm * w / m, dm])
        else:
            return np.array([0, 0, 0, 0, 0, 0, 0])

    return mass_accretion


# def mk_radius32(sigma_m32):
#     r_new = np.sqrt(sigma_m32**2/2*np.pi*4.302e-3*rhooo)
#     return r_new

def mk_mass_loss(m0_M31, r0_M31, m0_M32, r0_M32, sigma_v, *, dir=1, W=None):
    """
    A version of mass-loss function that computes dm/dt (=\dot{m})
    :param m0_M31:
    :param r0_M31:
    :param m0_M32:
    :param r0_M32:
    :param sigma_v:
    :param W:
    :return:
    """
    C_M32 = m0_M32 / r0_M32
    C_M31 = m0_M31 / r0_M31
    k_M32_M31 = np.sqrt(C_M32**3 / C_M31)
    k_pi = 4 / 3 * np.pi
    if W is None:
        W = r0_M32

    def rho(m, r):
        return 3 * m / (4 * np.pi * r**3)

    def rho_sph(c, r, w):
        """ Плотность толстой сферы """
        return c / (k_pi * (3 * r*r + w*w))

    def rho_m(c, m):
        """ Плотность от массы (для расчёта плотности M32)"""
        return c ** 3 / k_pi * m ** -2

    def mass(r, w):
        """ Масса M32 полученная иcходя из плотности в месте его пребывания в изотермич. сфере M31
        """
        return np.sqrt(C_M32**3 / C_M31) * np.sqrt(w**2 + 3*r**2)

    h = 3  # отношение плотностей, при кот. происходит "срыв". Это про кот. Сурдин говорил "3"

    def mass_loss(x, y, z, u, v, w, m32):
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)

        #         v_norm = np.sqrt(u*u + v*v + w*w)
        rho_M31 = rho_sph(C_M31, r, W)
        rho_M32 = rho_m(C_M32, m32)
        #         rho_m32 = rho(m32, Reff_m32)

        if m32 > 0 and rho_M32 < h * rho_M31:  # np.isclose(z, 0.0):
            # Знак <r,v> определяет направление движения отн. центра: если <r,v> < 0, тогда M32 движется в центр M31.
            # Если использовать $ <r,v> $, тогда срыв массы будет происходить на подлёте к центру.
            # Если использовать $ -<r,v> $, тогда срыв массы будет происходить при удалении от центра.
            # Логично, что $ -|<r,v>| $ приведёт к срыву массы в любом направлении
            #   пока выполняется условие $  rho_M32 < h*rho_M31 $.
            r_dot_v = -np.abs(u * x + v * y + w * z)
            #         k_M32_M31 = np.sqrt(C_M32**3 / C_M31)
            dm = 3*k_M32_M31 * r_dot_v / np.sqrt(h * (W**2 + 3*r2))
            return dir * dm
            # if dm < 0:
            #     return dm
            # else:
            #     return 0

        #             r_diff = -(x*u + y*v + z*w) / r
        #             w = 0
        #             return k_pi * r * r_diff / np.sqrt(k_pi*(w**2 + 3*r**2))
        else:
            return 0

    return mass_loss


def mk_mass_loss2(m0_M31, r0_M31, m0_M32, r0_M32, sigma_v, w=None):
    C_M32 = m0_M32 / r0_M32
    C_M31 = m0_M31 / r0_M31
    k_M32_M31 = np.sqrt(C_M32 ** 3 / C_M31)
    k_pi = 4 / 3 * np.pi
    if w is None:
        w = r0_M32

    def rho(m, r):
        return 3 * m / (4 * np.pi * r**3)

    def rho_sph(c, r, w):
        """ Плотность толстой сферы """
        return c / (k_pi * (3 * r*r + w*w))

    def rho_m(c, m):
        """ Плотность от массы (для расчёта плотности M32)"""
        return c ** 3 / k_pi * m ** -2

    def mass(r, w):
        """ Масса M32 полученная иcходя из плотности в месте его пребывания в изотермич. сфере M31
        """
        return np.sqrt(C_M32**3 / C_M31) * np.sqrt(w**2 + 3*r**2)

    h = 3  # отношение плотностей, при кот. происходит "срыв". Это про кот. Сурдин говорил "3"

    def mass_loss(x, y, z, u, v, w, m32):
        r2 = x * x + y * y + z * z
        r = np.sqrt(r2)

        #         v_norm = np.sqrt(u*u + v*v + w*w)
        rho_M31 = rho_sph(C_M31, r, w)
        rho_M32 = rho_m(C_M32, m32)
        #         rho_m32 = rho(m32, Reff_m32)

        if m32 > 0 and rho_M32 < h * rho_M31:  # np.isclose(z, 0.0):
            # Знак <r,v> определяет направление движения отн. центра: если <r,v> < 0, тогда M32 движется в центр M31.
            # Если использовать $ <r,v> $, тогда срыв массы будет происходить на подлёте к центру.
            # Если использовать $ -<r,v> $, тогда срыв массы будет происходить при удалении от центра.
            # Логично, что $ -|<r,v>| $ приведёт к срыву массы в любом направлении
            #   пока выполняется условие $  rho_M32 < h*rho_M31 $.
            r_dot_v = -np.abs(u * x + v * y + w * z)
            #         k_M32_M31 = np.sqrt(C_M32**3 / C_M31)
            dm = 3*k_M32_M31 * r_dot_v / np.sqrt(h * (w**2 + 3*r2))
            return dm
            # if dm < 0:
            #     return dm
            # else:
            #     return 0

        #             r_diff = -(x*u + y*v + z*w) / r
        #             w = 0
        #             return k_pi * r * r_diff / np.sqrt(k_pi*(w**2 + 3*r**2))
        else:
            return 0

    return mass_loss


@jit
def pericenter_massloss(mass_loss_fn, path_fn: DenseOutput):
    """
    Finds the point where body crosses the pericenter

    :param mass_loss_fn     A function used to estimate mass loss at the specific point.
    :type mass_loss_fn      f(x,y,z,vx,vy,vz,m) -> dm
    :param path_fn:         A function used to extrapolate the point on trajectory
    """

    def vnorm_f(s):
        """ Get the $ v_n $ - normal velocity.
        """
        # s: [rx, ry, rz, vx, vy, vz, m]
        R, v = s[0:3], s[3:6]
        return np.dot(R, v) / LA.norm(R)

    s0 = path_fn(path_fn.t_min)
    v0_norm = vnorm_f(s0)

    # Check if body (M32) falls within sphere around giant mass (M31)
    # if (LA.norm(s0[0:3]) < 500.0):
    #     return False

    s1 = path_fn(path_fn.t_max)
    v1_norm = vnorm_f(s1)

    # v_norm > 0 -- улетает, v_norm < 0 -- подлетает
    #  => sign(v0_norm) < sign(v1_norm) -- условие разворота, чтобы улететь вдаль
    if np.sign(v0_norm) < np.sign(v1_norm):
        # if np.isclose(v1_norm, 0.0, atol=4*np.finfo(float).eps):
        #     tz, sz = path_fn.t_max, s1
        # elif np.isclose(v0_norm, 0.0, atol=4*np.finfo(float).eps):
        #     tz, sz = path_fn.t_min, s0
        # else:
        tz = brentq(lambda t: vnorm_f(path_fn(t)), path_fn.t_min, path_fn.t_max, disp=True)
        sz = path_fn(tz)
        # Now tz equals the time point where body passes pericenter
        dm = mass_loss_fn(*sz.T)
        sz[6] += dm
        return tz, sz


class KinDyn:
    def __init__(self, fig: plt.Figure):
        self._fig = fig

    def plot_results(self, ax_loc, path, *, n=None, label=None, **plot_kwargs):
        if isinstance(ax_loc, tuple):
            ax = self._fig.add_subplot(*ax_loc, projection='3d')        # type: Axes3D
        else:
            assert isinstance(ax_loc, plt.Axes)
            ax = ax_loc

        ax.set_xlabel('X, kpc')
        ax.set_ylabel('Y, kpc')
        ax.set_zlabel('Z, kpc')

        # plt.contourf(X,Y,rho, levels=np.linspace(np.min(rho),np.max(rho),100), cmap='viridis')
        # plt.plot(path[:,0],path[:,1], c='r',lw=0.5)

        # n=240

        # *0.001 - x,y,z pc in kpc
        ax.plot(path[:n, 0] * 0.001, path[:n, 1] * 0.001, path[:n, 2] * 0.001,
                label=label, **plot_kwargs)
        return ax