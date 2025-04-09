from math import sin, cos, tan

mu = 398600 

def ode_orb3d(t, fullstate, data, TS=None):
    r, th, ph, u, v, w, m, lr, lth, lph, lu, lv, lw, lm = fullstate
    T_max, u_e, _ = data
    
    # switching function
    sf, LV = swfun(fullstate, data)
    if TS is None:
        T = T_max if sf > 0 else 0.0
    else:
        T = T_max if TS == 1 else 0.0
    
    au, av, aw = 0.0, 0.0, 0.0
    adu, adv, adw = 0.0, 0.0, 0.0
    if (T > 0.0) and (LV > 1e-12):
        au = (T/m) * (lu / LV) + adu
        av = (T/m) * (lv / LV) + adv
        aw = (T/m) * (lw / LV) + adw
    else:
        au = adu
        av = adv
        aw = adw
    
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
    dfdt[7] = (1/r**2) * (lth*v/cos(ph) + lph*w + lu*(-2*mu/r + v**2 + w**2) + lv*(-u*v + v*w*tan(ph)) + lw*(-u*w - v**2*tan(ph))) + ...
    dfdt[8] = ...
    dfdt[9] = 1/r*cos(phi)**2*(-lth*v*sin(ph) - lv*v*w + lw*v**2) + ...
    dfdt[10] = (1/r) * (-lr*r + lv*v + lw*w)
    dfdt[11] = (1/r) * (-lth/cos(ph) - 2*lu*v + lv*(u - w*tan(ph)) + 2*lw*v*tan(ph))
    dfdt[12] = (1/r) * (-lph - 2*lu*w - lv*v*tan(ph) + lw*u)
    dfdt[13] = (T*LV) / m**2 + ...
    
    return dfdt