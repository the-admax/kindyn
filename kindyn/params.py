import numpy as np

## ----------------------------------------------------------------------------------
## Параметры симуляции
##

## Units
# Mass:     1e6 Solar Masses
# Distance: Parsecs
# Time:     1e6 Years (Myr)

# Parameters
G = 4.302e3  # pc / M_sol * (km/s)^2
Gr = 4.302e-3
# G = 4490.0            # Gravitational Constant
n_steps = 500  # Number of steps in the simulation

# Initial conditions
t_init = 0  # Initial time of the simulation
t_final = -6000  # Final time of the simulation

dt = (t_final - t_init) / n_steps

m32_mass = 3e3  # [M_sol * 1e6], масса галактики m32
sigma_m32 = 65  # дисперсия скоростей в М32
r32_init = 500  # [pc] начальный радиус
# rho_m32_init = sigma_m32**2/(2*np.pi*4.302e-3*r32_init**2)# G в знаменателе приобретает свою изначальн.форму,если в ф-ле нет Msun
rho_m32_init = m32_mass / ((4 * np.pi / 3) * r32_init ** 3)

if 0:
    r_init = np.array([-20.84e3, -4.16e3, -7.9e3])  # m32 initial pos.
    #     r_init = np.array([100, 100, 110])
    v_init = np.array([-92.83859788982643, -18.53208096073308, 35.19313451677682])
#     v_init = np.array([ -155.36, -129.1, 58.84])
#     v_init = -0.001*(np.multiply(r_init, [1.0, 1.01, 1.0]))
if 1:
    r_init = np.array([-13.99523034, -4.25483436, -6.41132111]) * 1000
    v_init = np.array([-92.83859788982643, -18.53208096073308, 35.19313451677682])


# Halo component
v_h = 131.5  # [km/s], дисперсия скоростей
q = 1
r_c = 0.7e3  # [pc], радиус гало

# Gas component (Miyamoto-Nagai)
ra0_M31 = a = 35e3  # [pc] Радиус М31
b = 0.26e3  # [pc]
M = 6e5  # [M_sol * 1e6] Total mass of the galaxy
k_m31 = M / a

m_tot = M + m32_mass

# Dynamical Friction (Chandrasekhar)
logLambda = 1.0  # Coulomb logarithm
sigma = 181.0  # Velocity dispersion

v_s = 10  # скорость звука в газе из статьи

## ------------------------------------------------------------------------->8 ---