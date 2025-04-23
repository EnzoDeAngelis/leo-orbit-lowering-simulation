import numpy as np
from pymsis import run_msis, msis_inputs
from datetime import datetime, timedelta


def spherical_to_cartesian(r, th, ph):
    """
    Converte coordinate sferiche (r, theta, phi) → cartesiane (x, y, z)
    θ = longitudine, φ = latitudine. Angoli in radianti.
    """
    x = r * np.cos(ph) * np.cos(th)
    y = r * np.cos(ph) * np.sin(th)
    z = r * np.sin(ph)
    return np.array([x, y, z])


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

def atmosphere_velocity_ijk(r_ijk, R = 6371, V = 7.909788019132536):
    """
    Velocità adimensionale dell'atmosfera in EME2000.
    """
    t_scale = R/V
    omega_earth = np.array([0.0, 0.0, 7.2921150e-5]) * t_scale   # Il prodotto per t_scale rende omega adimensionale
    return np.cross(omega_earth, r_ijk)


def get_density_msis(alt, lon, lat, date = datetime(2025, 4, 15, 12, 0, 0), f107s=150.0, f107as=150.0, ap=10):

    """
    Calcola la densità atmosferica a una certa altitudine usando il modello MSISE-00.
    
    Parametri:
    - date: oggetto datetime (data e ora in UTC) 
    - lon: longitudine in gradi
    - lat: latitudine in gradi
    - alt: altitudine in km
    - f107s: (default=150.0) flusso solare F10.7 (I valori di solito spaziano da 70 a 300)
    - f107as: (default=150.0) media di 81 giorni di F10.7
    - ap: (default=10) indice geomagnetico Ap (I valori di solito spaziano da 0 a 20)

    Restituisce:
    - densità atmosferica in kg/m³ (valore float)
    """
    
    # Create MSIS input object
    inputs = msis_inputs(
        times=[date],
        altitudes=[alt],
        latitudes=[lat],
        longitudes=[lon],
        f107s=[f107s],
        f107as=[f107as],
        aps=[ap]
    )

    output = run_msis(inputs)
    rho = output['Total mass density'][0]  # [kg/m³]
    return rho

def compute_drag_acceleration_ijk(rho, v_rel, m, Cd = 2.2, A = 0.5):
    """
    Calcola la forza di drag in base cartesiana inerziale (ijk).
    """
    v_rel_norm = np.linalg.norm(v_rel)
    v_hat = v_rel / v_rel_norm  # versore della velocità relativa
    a_drag = -0.5 * rho * (Cd * A / m) * v_rel_norm**2 * v_hat

    return a_drag


def compute_drag_acceleration_zen(r, th, ph, u, v, w, m, CD, A, R = 6371):
    
    r_ijk = spherical_to_cartesian(r, th, ph)
    T_zen_to_ijk = zen_to_ijk_matrix(th, ph)
    v_zen = np.array[u, v, w]
    v_ijk = T_zen_to_ijk @ v_zen
    
    # Velocità dell’atmosfera in EME2000
    v_atm = atmosphere_velocity_ijk(r_ijk)
    # Velocità relativa satellite - atmosfera
    v_rel = v_ijk - v_atm
    
    # Coordinate per MSISE
    alt = (r*R - R)      # altitudine rispetto al suolo in km
    lon = np.rad2deg(th)
    lat = np.rad2deg(ph)
    
    
    # Calcolo la densità atmosferica (MSISE-00)
    # Altitudine in km ed angoli in gradi
    rho = get_density_msis(alt, lon, lat)
    
    # Calcolo accelerazione dovuta alla resistenza atmosferica
    a_drag_ijk = compute_drag_acceleration_ijk(rho, v_rel, m)
    
    # Proietto la drag sulla terna ZEN
    a_drag_zen = T_zen_to_ijk.T @ a_drag_ijk

    return a_drag_zen