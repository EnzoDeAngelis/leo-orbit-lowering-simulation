from numba import njit
import numpy as np

@njit
def spherical_to_cartesian(r, th, ph):
    """
    Converte coordinate sferiche (r, theta, phi) → cartesiane (x, y, z)
    θ = longitudine, φ = latitudine. Angoli in radianti.
    """
    x = r * np.cos(ph) * np.cos(th)
    y = r * np.cos(ph) * np.sin(th)
    z = r * np.sin(ph)
    return np.array([x, y, z])


@njit
def zen_to_ijk_matrix(th, ph):
    """
    Costruisce la matrice di rotazione che converte da base ZEN a EME2000 cartesiana.
    Angoli in radianti.
    """
    # Zenith (verso l'esterno): r-hat
    z_hat = np.array([
        np.cos(ph) * np.cos(th),
        np.cos(ph) * np.sin(th),
        np.sin(ph)
    ])
    
    # East: direzione crescente di theta
    e_hat = np.array([
        -np.sin(th),
        np.cos(th),
        0.0
    ])

    # North: tangente a phi
    n_hat = np.array([
        -np.sin(ph) * np.cos(th),
        -np.sin(ph) * np.sin(th),
        np.cos(ph)
    ])

    # Ogni vettore è una colonna: T @ v_ZEN = v_EME2000
    T = np.column_stack((z_hat, e_hat, n_hat)) 
    return T


@njit
def ijk_to_zen_matrix(th, ph):
    v1 = np.array([np.cos(ph)*np.cos(th), -np.sin(th), -np.sin(ph)*np.cos(th)])
    v2 = np.array([np.cos(ph)*np.sin(th), np.cos(th), -np.sin(ph)*np.sin(th)])
    v3 = np.array([np.sin(ph), 0, np.cos(ph)])
    
    T = np.column_stack((v1, v2, v3))
    return T