import numpy as np
from scipy.integrate import solve_ivp
from OCP.functions import ode_orb3d, swfun
import yaml
import os

R = 3396
V = 3.551255605022341
T = R/V
A = V**2/R
M = 1000
TN = M * A

# === Condizioni iniziali adimensionali ===
if os.path.exists("problems/prob.yaml"):
    with open("problems/prob.yaml", "r") as f:
        yaml_data = yaml.safe_load(f)
    states_order = ["r", "theta", "phi", "u", "v", "w", "m"]
    _initial_state = [yaml_data["initial_state"][key] for key in states_order]
else:
    _initial_state = [1.2, 0.0, 0.0, 0.0, np.sqrt(1/1.2), 0.0, 1.0]
fullstate = np.append(_initial_state, np.zeros(7))

# === Parametri ===
T = 0.0
u_e   = 40000.0/(V*10**3)
delta = 1e-12
data = [T, u_e, delta]
sw, LV = swfun(fullstate, data)
thr = None

# === Calcolo del tempo finale adimensioanle ===
T_orbita = np.sqrt(fullstate[0] ** 3)
t_final = 2 * T_orbita

# === Integrazione ===
sol = solve_ivp(
    lambda t, y: ode_orb3d(t, y, data, thr),
    (0, t_final),
    fullstate,
    method="RK45",
    t_eval=[t_final]  # solo il punto finale
)

# === Estrazione degli stati finali ===
r, theta, phi, u, v, w = sol.y[:6, -1]

print(f"Stati iniziali: {fullstate}")
print("Stati finali dopo 2 orbite:")
print(f"r     = {r:.16f}")
print(f"theta = {theta:.16f}")
print(f"phi   = {phi:.16f}")
print(f"u     = {u:.16f}")
print(f"v     = {v:.16f}")
print(f"w     = {w:.106f}")
