import math

R = 6371     # Raggio terrestre
mu = 398600  # Mu Terra

# Quota di riferimento
h = 498.8

# Raggio adimensionale
r = (h+R)/R

v = math.sqrt(mu/(r*R))  # Velocità dimensionale alla quota h
vc1 = math.sqrt(mu/R)    # Valore per adimensionalizzare la velocità
v_nd = v/vc1             # Velocità adimensionale alla quota h

print(f"Il raggio adimensionale vale r={r}")
print(f"La velocità adimensionale alla quota h={h} da inserire in prob.yaml è {v_nd}")
