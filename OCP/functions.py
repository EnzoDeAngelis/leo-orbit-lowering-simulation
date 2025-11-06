import numpy as np
from datetime import datetime, timedelta
import pymsis
import spiceypy as spice
from numba import njit
from math import sin, cos, tan, sqrt, asin, acos, pi


# ------------------------------------------------------
# FUNZIONI UTILIZZATE
# ------------------------------------------------------
def calculate_rho_space(r, theta, phi, date=datetime(2025, 3, 15, 12), f107s=150.0, f107as=150.0, ap=10):
    """
    Calcola la densità atmosferica a una certa posizione spaziale (r, theta, phi)
    usando il modello MSISE-00, convertendo prima in coordinate geografiche.

    Parametri:
    - r      : distanza dal centro della Terra (km)
    - theta  : longitudine siderale (radianti, [0, 2pi])
    - phi    : latitudine geocentrica (radianti, [-pi/2, pi/2])
    - date   : datetime in UTC (default: 15 marzo 2025, 12:00)
    - f107s  : indice solare F10.7 istantaneo
    - f107as : media di 81 giorni di F10.7
    - ap     : indice geomagnetico Ap (singolo valore)

    Ritorna:
    - densità atmosferica (kg/m^3)
    """
    # Converte coordinate sferiche (r, theta, phi) in coordinate geografiche (lon, lat, alt)
    lon, lat, alt = spherical_j2000_to_geo_no_spice(r, theta, phi, date)

    # MSISE richiede array numpy anche per singole richieste
    dates = np.array([date])
    f107s = np.array([f107s])
    f107as = np.array([f107as])
    aps = np.array([[ap] * 7])  # vettore Ap da 7 valori richiesto da MSISE

    lons = np.array([lon])
    lats = np.array([lat])
    alts = np.array([alt])

    # Calcolo densità con pymsis
    results = pymsis.calculate(dates, lons, lats, alts, f107s, f107as, aps, version=2.1)

    # Estrazione densità di massa (kg/m^3)
    rho = results[0, pymsis.Variable.MASS_DENSITY]
    return rho

# ------------------------------------------------------
def spherical_j2000_to_geo_no_spice(r, theta, phi, date):
    """
    Converte coordinate sferiche geocentriche (r, theta, phi) in longitudine, latitudine e altitudine
    geografiche approssimate, usando una formula del tempo siderale di Greenwich (GMST).
    Questa funzione serve perchè PYMSIS prende in imput: lat long e alt.

    Parametri:
    - r      : raggio geocentrico (km)
    - theta  : angolo orario RA (radianti)
    - phi    : declinazione = latitudine geocentrica (radianti)
    - date   : datetime UTC

    Ritorna:
    - lon_deg : longitudine geografica (gradi, [-180, 180])
    - lat_deg : latitudine geocentrica (gradi, [-90, 90])
    - alt_km  : altitudine sopra la superficie (km)
    """
    # Calcolo Julian Date
    Y, M, D = date.year, date.month, date.day
    h = date.hour + date.minute/60 + date.second/3600
    if M <= 2:
        Y -= 1
        M += 12
    A = np.floor(Y / 100)
    B = 2 - A + np.floor(A / 4)
    JD0 = np.floor(365.25 * (Y + 4716)) + np.floor(30.6001 * (M + 1)) + D + B - 1524.5
    JD = JD0 + h / 24.0

    # Tempo siderale di Greenwich (GMST) in radianti
    T = (JD - 2451545.0) / 36525.0
    GMST_deg = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + 0.000387933 * T**2 - T**3 / 38710000.0
    GMST_deg = GMST_deg % 360.0
    GMST_rad = np.deg2rad(GMST_deg)

    # Conversione
    lon_rad = (theta - GMST_rad + np.pi) % (2 * np.pi) - np.pi  # normalizzazione [-pi, pi]
    lat_rad = phi
    alt_km = r - 6371.0  # raggio medio terrestre

    return np.degrees(lon_rad), np.degrees(lat_rad), alt_km


def calcolare_rho_matrix_4d(start_date):
    t_vals = np.linspace(0, 3600*24*1, 5) # Se cambio queste righe devo cambiare anche le righe di calcolo di interpolate
    r_vals = np.linspace(6600, 7000, 30)
    theta_vals = np.linspace(0, 2*np.pi, 37)
    phi_vals = np.linspace(-np.pi/2, np.pi/2, 19)

    rho_matrix = np.zeros((len(t_vals), len(r_vals), len(phi_vals), len(theta_vals)))

    for it, t_sec in enumerate(t_vals):
        current_date = start_date + timedelta(seconds=float(t_sec))
        for i, r in enumerate(r_vals):
            for j, phi in enumerate(phi_vals):
                for k, theta in enumerate(theta_vals):
                    rho_matrix[it, i, j, k] = calculate_rho_space(r, theta, phi, current_date)

    return rho_matrix

@njit
def spherical_to_cartesian(r, lon, lat):
    """
    Converte coordinate sferiche (r, lon, lat) → cartesiane (x, y, z).
    - lon: longitudine [rad] (0 a 2π)
    - lat: latitudine [rad] (-π/2 a π/2)
    """
    x = r * cos(lat) * cos(lon)
    y = r * cos(lat) * sin(lon)
    z = r * sin(lat)
    return np.array([x, y, z])


@njit
def zen_to_ijk_matrix(lon, lat):
    """
    Matrice di rotazione ZEN (Zenith-East-North) → IJK (EME2000).
    - lon: longitudine [rad]
    - lat: latitudine [rad]
    """
    # Zenith (radial outward)
    z_hat = np.array([
        cos(lat) * cos(lon),
        cos(lat) * sin(lon),
        sin(lat)
    ])
    
    # East (along-track per orbite equatoriali)
    e_hat = np.array([
        -sin(lon),
        cos(lon),
        0.0
    ])

    # North (cross-track)
    n_hat = np.array([
        -sin(lat) * cos(lon),
        -sin(lat) * sin(lon),
        cos(lat)
    ])

    # Ogni vettore è una colonna: T @ v_ZEN = v_EME2000
    T = np.column_stack((z_hat, e_hat, n_hat)) 
    return T
