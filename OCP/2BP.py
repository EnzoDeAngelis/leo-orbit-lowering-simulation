from math import sin, cos, tan
import numpy as np
from drag_functions import compute_drag_acceleration_zen
from datetime import datetime
from compute_derivative import compute_drag_partials_extended
 

def swfun(fullstate, data):
    r, th, ph, u, v, w, m, lr, lth, lph, lu, lv, lw, lm = fullstate
    T_max, u_e, _ = data
    
    LV = np.sqrt(lu**2 + lv**2 + lw**2)
    return LV - lm*m/u_e, LV

def ode_orb3d(t, fullstate, data, TS=None):
    r, th, ph, u, v, w, m, lr, lth, lph, lu, lv, lw, lm = fullstate
    T_max, u_e, _ = data
    mu = 398600
    
    # switching function
    sf, LV = swfun(fullstate, data)
    if TS is None:
        T = T_max if sf > 0 else 0.0
    else:
        T = T_max if TS == 1 else 0.0
    
    au, av, aw = 0.0, 0.0, 0.0
    
    a_drag_zen = compute_drag_acceleration_zen(r, th, ph, u, v, w, m)
    
    a_drag_u = a_drag_zen[0]
    a_drag_v = a_drag_zen[1]
    a_drag_w= a_drag_zen[2]
    
    
    if (T > 0.0) and (LV > 1e-12):
        au = (T/m) * (lu / LV) + a_drag_u
        av = (T/m) * (lv / LV) + a_drag_v
        aw = (T/m) * (lw / LV) + a_drag_w
    else:
        au = a_drag_u
        av = a_drag_v
        aw = a_drag_w
    
    # Calcolo derivate che compaiono nelle ODE
    J = compute_drag_partials_extended(r, th, ph, u, v, w, m)
    # Estrai righe: una per ogni componente dell'accelerazione (u, v, w)
    da_u_dr, da_u_dth, da_u_dph, da_u_dm = J[0, :]
    da_v_dr, da_v_dth, da_v_dph, da_v_dm = J[1, :]
    da_w_dr, da_w_dth, da_w_dph, da_w_dm = J[2, :]
    
    dfdt = np.zeros_like(fullstate)    
    
    # States
    dfdt[0] = u
    dfdt[1] = v/(r*cos(ph))
    dfdt[2] = w/r
    dfdt[3] = -mu/r**2 + v**2/r + w**2/r + au
    dfdt[4] = -u*v/r + v*(w/r) * tan(ph)+ av
    dfdt[5] = -u*w/r - (v**2/r) * tan(ph) + aw
    dfdt[6] = -T / u_e

    # Costates
    dfdt[7] = (1/r**2) * (lth*v/cos(ph) + lph*w + lu*(-2*mu/r + v**2 + w**2) + lv*(-u*v + v*w*tan(ph)) + lw*(-u*w - v**2*tan(ph))) - lu*da_u_dr - lv*da_v_dr - lw*da_w_dr
    dfdt[8] = - lu*da_u_dth - lv*da_v_dth - lw*da_w_dth
    dfdt[9] = 1/(r*cos(ph)**2)*(-lth*v*sin(ph) - lv*v*w + lw*v**2) - lu*da_u_dph - lv*da_v_dph - lw*da_w_dph
    dfdt[10] = (1/r) * (-lr*r + lv*v + lw*w)
    dfdt[11] = (1/r) * (-lth/cos(ph) - 2*lu*v + lv*(u - w*tan(ph)) + 2*lw*v*tan(ph))
    dfdt[12] = (1/r) * (-lph - 2*lu*w - lv*v*tan(ph) + lw*u)
    dfdt[13] = (T*LV) / m**2 - lu*da_u_dm - lv*da_v_dm -lw*da_w_dm
    
    return dfdt

def compute_H_orb3d(fullstate, data, TS=None):
    r, th, ph, u, v, w, m, lr, lth, lph, lu, lv, lw, lm = fullstate
    T_max, u_e, _ = data
    mu = 398600
    
    sf, LV = swfun(fullstate, data)
    
    if TS is None:
        T = T_max if sf > 0 else 0.0
    else:
        T = T_max if TS == 1 else 0.0
        
    au, av, aw = 0.0, 0.0, 0.0
    
    a_drag_zen = compute_drag_acceleration_zen(r, th, ph, u, v, w, m)
    
    a_drag_u = a_drag_zen[0]
    a_drag_v = a_drag_zen[1]
    a_drag_w= a_drag_zen[2]
    
    if (T > 0.0) and (LV > 1e-12):
        au = (T/m) * (lu / LV) + a_drag_u
        av = (T/m) * (lv / LV) + a_drag_v
        aw = (T/m) * (lw / LV) + a_drag_w
    else:
        au = a_drag_u
        av = a_drag_v
        aw = a_drag_w
        
    return(
        lr*u + lth*v/(r*cos(ph)) + lph*w/r + 
        lu*(-mu/r**2 + v**2/r + w**2/r + au) + 
        lv*(-u*v/r + v*(w/r)*tan(ph) + av) + 
        lw*(-u*w/r - (v**2/r)*tan(ph) + aw)
        - lm*T/u_e
    )