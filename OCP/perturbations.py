import numpy as np
from numba import njit
from math import sin, cos, tan, sqrt, asin, acos, pi
from functions import zen_to_ijk_matrix, spherical_to_cartesian


# ------------------------------ parametri -------------------------------
R_Earth     = 6371.0            # [km]
mi_Earth    = 398600.0          # [km^3/s^2]
omega_Earth = 7.2921159e-5      # [rad/s]
Cd          = 2.2               # coefficiente di resistenza [-]
A_drag      = 1.0               # superficie [m^2]
M           = 500               # massa spacecraft [kg]
v_Earth     = sqrt(mi_Earth / R_Earth)      # [km/s]
a_c         = (v_Earth**2 / R_Earth) * 1000 # accelerazione da [km/s^2] a [m/s^2]


#------------------------------- definizione varie funzioni ---------------------------------
@njit
def compute_total_perturbations(r, th, ph, u, v, w, m, rho_val):
    
    a_total = compute_drag_acceleration_zen(r, th, ph, u, v, w, m, rho_val)
    
    return a_total
##############################     RESISTENZA ATMOSFERICA     ##############################


# calcolo contributo dovuto alla resistenza atmosferica in ijk in [m/s^2]
@njit
def compute_drag_acceleration_ijk(rho_val, v_rel, mass):
    v_rel_norm = np.linalg.norm(v_rel) # velocità passata in input in [m/s] e mantenuta in [m/s]
    if v_rel_norm == 0:
        return np.zeros(3)
    a_drag = -0.5 * rho_val * (Cd * A_drag / mass) * v_rel_norm * v_rel # accelerazione restituita in [m/s^2]
    return a_drag


# calcolo prodotto vettoriale tra velocità di rotazione terrestre e raggio in ijk
@njit
def atmosphere_velocity_ijk(r_ijk):
    """
    Velocità dell'atmosfera dovuta alla rotazione terrestre.
    omega = [0, 0, omega_earth]
    """
    omega_earth = np.array([0.0, 0.0, omega_Earth]) # velocità angolare in [rad/s]
    return np.cross(omega_earth, r_ijk) # prodotto vettoriale tra [rad/s] e [m] --> [m/s]


# calcolo contributo dovuto alla resistenza atmosferica in coordinate zen
@njit
def compute_drag_acceleration_zen(r, th, ph, u, v, w, m, rho_val):
    
    r_ijk = spherical_to_cartesian(r, th, ph) * (R_Earth * 1000) # raggio calcolato a partire dal valore adimensionale di r e convertito poi in [m]
    T_zen_to_ijk = zen_to_ijk_matrix(th, ph)
    v_zen = np.array([u, v, w]) * (v_Earth * 1000) # valocità calcolata a partire dai valori adimensionali di u, v e w e convertito poi in [m/s]
    v_ijk = T_zen_to_ijk @ v_zen

    v_atm = atmosphere_velocity_ijk(r_ijk) # restituisce la velocità in [m/s]
    v_rel = v_ijk - v_atm # differenza tra velocità in [m/s]

    # rho_val = 1e-12 # densità atmosferica in [kg/m^3] (valore fittizio, da sostituire con modello atmosferico)
    mass = m * M # massa dimensionalizzata [kg]

    a_drag_ijk = compute_drag_acceleration_ijk(rho_val, v_rel, mass)

    a_drag_zen = T_zen_to_ijk.T @ a_drag_ijk
    
    a_drag_zen /= a_c

    return a_drag_zen

#################### DERIVATE PARTIALS OF DRAG ACCELERATION ####################
delta = 1e-8

# Funzione helper esterna per compatibilità Numba
@njit
def perturbation_helper(r, th, ph, u, v, w, m, rho_val):
    return compute_total_perturbations(r, th, ph, u, v, w, m, rho_val)

@njit
def compute_total_derived_perturbations(r, th, ph, u, v, w, m, rho_val):
    # derivata rispetto a r
    a_plus = perturbation_helper(r + delta, th, ph, u, v, w, m, rho_val)
    a_minus = perturbation_helper(r - delta, th, ph, u, v, w, m, rho_val)
    da_dr = (a_plus - a_minus) / (2 * delta)

    # derivata rispetto a theta
    a_plus = perturbation_helper(r, th + delta, ph, u, v, w, m, rho_val)
    a_minus = perturbation_helper(r, th - delta, ph, u, v, w, m, rho_val)
    da_dtheta = (a_plus - a_minus) / (2 * delta)

    # derivata rispetto a phi
    a_plus = perturbation_helper(r, th, ph + delta, u, v, w, m, rho_val)
    a_minus = perturbation_helper(r, th, ph - delta, u, v, w, m, rho_val)
    da_dphi = (a_plus - a_minus) / (2 * delta)

    # derivata rispetto a massa m
    a_plus = perturbation_helper(r, th, ph, u, v, w, m + delta, rho_val)
    a_minus = perturbation_helper(r, th, ph, u, v, w, m - delta, rho_val)
    da_dm = (a_plus - a_minus) / (2 * delta)

    # Costruzione matrice Jacobiana 3x4
    J = np.zeros((3, 4))
    J[:, 0] = da_dr
    J[:, 1] = da_dtheta
    J[:, 2] = da_dphi
    J[:, 3] = da_dm

    return J


##############################     PRESSIONE SOLARE     ##############################

# Funzione per calcolare l'accelerazione perturbativa dovuta alla pressione solare (SRP)
# def srp_acceleration_spherical(r_km, theta, phi, r_sun_eci,
#                                 mass_kg, surface_m2, eta_R=0.7, t=0, epoch=datetime(2025, 3, 21, 12)):
#     """
#     Calcola l'accelerazione perturbativa dovuta alla pressione solare (SRP)
#     in coordinate sferiche (u,v,w) nel sistema EME2000.

#     Parametri:
#     - r_km       : distanza satellite-Terra [km]
#     - theta, phi : coordinate sferiche del satellite [rad]
#     - r_sun_eci  : vettore Sole→Terra in ECI (J2000) [km, array (3,)]
#     - mass_kg    : massa del satellite [kg]
#     - surface_m2 : area esposta [m²]
#     - eta_R      : coefficiente di riflettività (default: 0.7)

#     Ritorna:
#     - (a_u, a_v, a_w): componenti dell'accelerazione SRP [km/s²]
#     """

#     # Test eclissi
#     if is_in_eclipse_spice(r_km, theta, phi, t, epoch):
#         return 0.0, 0.0, 0.0

#     # Costanti
#     AU_km = 149597870.7               # km
#     p0 = 4.56e-6                      # N/m², pressione a 1 AU
#     c = 2.99792458e8                 # m/s (non necessario se p0 usato direttamente)

#     # Distanza Sole-satellite in AU
#     d_sun_km = np.linalg.norm(r_sun_eci)
#     d_sun_au = d_sun_km / AU_km

#     # Pressione alla distanza attuale
#     p = p0 / (d_sun_au ** 2)         # N/m²

#     # Accelerazione scalare totale [m/s²]
#     Gamma = (1 + eta_R) * p * surface_m2 / mass_kg   # m/s²

#     # Conversione in km/s²
#     Gamma_km = Gamma / 1000.0

#     # Versori sferici nel punto (theta, phi)
#     e_r = np.array([
#         np.cos(theta) * np.cos(phi),
#         np.sin(theta) * np.cos(phi),
#         np.sin(phi)
#     ])
#     e_theta = np.array([
#         -np.sin(theta),
#          np.cos(theta),
#          0.0
#     ])
#     e_phi = np.array([
#         -np.cos(theta) * np.sin(phi),
#         -np.sin(theta) * np.sin(phi),
#          np.cos(phi)
#     ])

#     # Vettore satellite → Sole (cioè -r_sun_eci)
#     to_sun_vec = -r_sun_eci

#     # Proiezione sulle direzioni locali (radiale, θ, φ)
#     r_es_u = np.dot(to_sun_vec, e_r)
#     r_es_v = np.dot(to_sun_vec, e_theta)
#     r_es_w = np.dot(to_sun_vec, e_phi)

#     # Accelerazione sferica in km/s²
#     a_u = -Gamma_km / r_km**3 * (r_es_u - r_km)
#     a_v = -Gamma_km / r_km**3 * r_es_v
#     a_w = -Gamma_km / r_km**3 * r_es_w

#     return a_u, a_v, a_w