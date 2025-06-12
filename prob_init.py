import math
import numpy as np

R = 3396     # Raggio Marte
mu = 4.282837e4  # Mu Marte
T = 0.05 # N
m = 1000 # kg

# Quota di riferimento
h = 195

# Raggio adimensionale
a = (h+R)/R

v = math.sqrt(mu/(a*R))  # Velocità dimensionale alla quota h
vc1 = math.sqrt(mu/R)    # Valore per adimensionalizzare la velocità
v_nd = v/vc1             # Velocità adimensionale alla quota h

delta = 1 # Spingo per tutta l'orbita
delta_a = delta*(4*np.pi/mu)*(a*R)**3*(T/m)

print(f"Il raggio adimensionale vale r={a}")
print(f"La velocità adimensionale alla quota h={h} da inserire in prob.yaml è {v_nd}")
print(f"Alla quota di {h} km posso scendere massimo di {delta_a} m per orbita")
