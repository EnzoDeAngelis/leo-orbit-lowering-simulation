# Calcolo le derivate delle componenti della drag (au,av,aw) rispetto a r,theta,phi,m col metodo delle differenze finite centrali
import numpy as np
from drag_functions import compute_drag_acceleration_zen
from numba import njit

@njit
def compute_drag_partials_extended(r, th, ph, u, v, w, m, CD = 2.2, A = 0.5, delta=1e-5):
    # Funzione centrale
    def f(r_, th_, ph_, m_):
        return compute_drag_acceleration_zen(r_, th_, ph_, u, v, w, m_, CD, A)

    # Derivata rispetto a r
    a_plus = f(r + delta, th, ph, m)
    a_minus = f(r - delta, th, ph, m)
    da_dr = (a_plus - a_minus) / (2 * delta)

    # Derivata rispetto a theta
    a_plus = f(r, th + delta, ph, m)
    a_minus = f(r, th - delta, ph, m)
    da_dth = (a_plus - a_minus) / (2 * delta)

    # Derivata rispetto a phi
    a_plus = f(r, th, ph + delta, m)
    a_minus = f(r, th, ph - delta, m)
    da_dph = (a_plus - a_minus) / (2 * delta)

    # Derivata rispetto a massa m
    a_plus = f(r, th, ph, m + delta)
    a_minus = f(r, th, ph, m - delta)
    da_dm = (a_plus - a_minus) / (2 * delta)
    
    # Matrice Jacobiana 3x4: colonne = r, th, ph, m
    J = np.column_stack([da_dr, da_dth, da_dph, da_dm])

    return J