# perturbations.py

import numpy as np
from numba import njit
from OCP.rotation_matrix import zen_to_ijk_matrix, spherical_to_cartesian


def load_density_data(filepath='mars_output.txt'):
    """
    Carica dati di densità atmosferica da file.
    """
    data = np.loadtxt(filepath, skiprows=1)
    altitude = data[:, 1]
    density = data[:, 4]
    return altitude, density


@njit
def linear_interp(x, xp, fp):
    """
    Interpolazione lineare compatibile con Numba.
    """
    if x <= xp[0]:
        return fp[0]
    elif x >= xp[-1]:
        return fp[-1]

    for i in range(len(xp) - 1):
        if xp[i] <= x < xp[i + 1]:
            slope = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i])
            return fp[i] + slope * (x - xp[i])
    
    return fp[-1]  # fallback


@njit
def compute_drag_acceleration_ijk(rho, v_rel, m, Cd=2.2, A=1):
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm == 0:
        return np.zeros(3)
    v_hat = v_rel / v_rel_norm
    a_drag = -0.5 * rho * (Cd * A / m) * v_rel_norm**2 * v_hat
    return a_drag


@njit
def atmosphere_velocity_ijk(r_ijk, R=3396, V=3.551240265):
    t_scale = R / V
    omega_mars = np.array([0.0, 0.0, 7.088e-5]) * t_scale
    return np.cross(omega_mars, r_ijk)


@njit
def compute_drag_acceleration_zen(r, th, ph, u, v, w, m, altitude_array, density_array, R=3396, Cd=2.2, A=1): 
    r_ijk = spherical_to_cartesian(r, th, ph) 
    T_zen_to_ijk = zen_to_ijk_matrix(th, ph)
    v_zen = np.array([u, v, w])
    v_ijk = T_zen_to_ijk @ v_zen

    v_atm = atmosphere_velocity_ijk(r_ijk)
    v_rel = v_ijk - v_atm

    alt = r * R - R  # altitudine in km
    rho_val = linear_interp(alt, altitude_array, density_array)

    a_drag_ijk = compute_drag_acceleration_ijk(rho_val, v_rel, m, Cd, A)
    
    if r <= 1.0574204946996466:
        a_drag_zen = T_zen_to_ijk.T @ a_drag_ijk
    else:
        a_drag_zen = np.zeros(3)
        
    return a_drag_zen
