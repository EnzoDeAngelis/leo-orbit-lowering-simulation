import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
import yaml
import os
from math import sin, cos, tan
from OCP.perturbations import compute_drag_acceleration_zen, compute_total_derived_perturbations

from problems.base_problem import BaseProblem

class OrbitalProblem(BaseProblem):
    def __init__(self, initial_state=[1.2, 0.0, 0.0, np.sqrt(1/1.2), 1],     # Lo stato iniziale viene letto da qui solo se non esiste il file prob.yaml
                 switching_structure=[1,0,1]):
        super().__init__()
        self.initial_state = initial_state
        self.boundary_conditions = None
        self.switching_structure = switching_structure
        self.parse_yaml()
        print(self.initial_state)
        print(self.boundary_conditions)
        # store custom defaults if needed


    def parse_yaml(self):
        # if exists prob.yaml
        if not os.path.exists("problems/prob.yaml"):
            return
        with open("problems/prob.yaml", "r") as f:
            data = yaml.safe_load(f)

        # in data["initial_state"] there are key-value pairs
        # the key is the variable and has to match the states ones
        # the value is the value of the variable

        _initial_state = data["initial_state"]
        self.initial_state = [_initial_state[x] for x in self.states()]
        self.boundary_conditions = data["boundary_conditions"]

    def name(self):
        return "Orbital"

    def default_param_guess(self):
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 3.500311]  # esempio default guess
    
    ##### DEVE ESSERE MODIFICATO DA PARTE VOSTRA IN BASE AL PROBLEMA #####
    def states(self):
        return ["r", "theta", "phi", "u", "v", "w", "m"]
    
    ##### DEVE ESSERE MODIFICATO DA PARTE VOSTRA IN BASE AL PROBLEMA #####
    def costates(self):
        return ["lr", "ltheta", "lphi", "lu", "lv", "lw", "lm"] 
    
    def state_labels(self):
        return [x + "(t)" for x in self.states()]
    
    def costate_labels(self):
        return ["\lambda " + x + "(t)" for x in self.states()]

    def ode_func(self, t, state, data, thr=None):
        return ode_orb3d(t, state, data, thr)

    def compute_H(self, t, state, data, thr=None):
        return compute_H_orb3d(t, state, data, thr)
    
    def compute_sf(self, state, data):
        return swfun(state, data)

    def boundary_error(self, param_guess, data, timefree, tols=1e-6, method="RK45"):
        lr0, lth0, lph0, lu0, lv0, lw0, lm0, *tfs = param_guess                # A prescindere da quanti archi si abbiano, tutti i tempi di switch 
                                                                               # ed il tempo finale vengono salvati nel vettore tfs (*tfs)
        full0 = [*self.initial_state, lr0, lth0, lph0, lu0, lv0, lw0, lm0]

        NA = 1
        if self.switching_structure is not None:
            NA = len(self.switching_structure)
        
        err = np.array([])
        tstart = 0.0

        for arc_idx in range(len(tfs)):
            t_end = tfs[arc_idx]
            # Se i tempi ai bordi combaciano, la soluzione ha "eliminato" l'arco
            if tstart == t_end:
                continue
            thr = None
            if self.switching_structure is not None:
                thr = self.switching_structure[arc_idx]
            sol = solve_ivp(
                lambda t, s: self.ode_func(t, s, data, thr),
                            (tstart, t_end), full0, 
                            dense_output=True,
                            method=method,
                            atol=tols, rtol=tols,
                )
            rf, thf, phf, uf, vf, wf, mf, lrf, lthf, lphf, luf, lvf, lwf, lmf = sol.y[:, -1]
        
            sf = np.sqrt(
                sol.y[10,:]**2 + sol.y[11,:]**2 + sol.y[12,:]**2) - \
                    sol.y[-1,:]*sol.y[6,:]/data[1]
            
            # Se il problema è a tempo libero e siamo all'ultimo arco,
            # calcoliamo l'Hamiltoniana finale
            if timefree and arc_idx == NA-1:
                Hf = self.compute_H(sol.t[-1], sol.y[:, -1], data, thr)
                # Questo moltiplicatore aiuta la convergenza del tempo
                if abs(Hf) > 1e-4:
                    Hf *= 100
                elif abs(Hf) > 1e-5:
                    Hf *= 10
                err = np.append(err, Hf*100)
            if arc_idx != NA-1:
                err = np.append(err, np.array([sf[-1]]))
            
            # Aggiorna le condizioni iniziali per il prossimo arco
            full0 = sol.y[:, -1]
            tstart = t_end

        ##### LETTURA AUTOMATICA ERRORI "SEMPLICI" #####
        for k, v in self.boundary_conditions.items():
            if k == "r":
                err = np.append(err, [(rf - v)])
            elif k == "theta":
                err = np.append(err, [thf - v])
            elif k == "phi":
                err = np.append(err, [phf - v])    
            elif k == "u":
                err = np.append(err, [uf - v])
            elif k == "v":
                err = np.append(err, [vf - v])
            elif k == "w":
                err = np.append(err, [wf - v])
            elif k == "m":
                err = np.append(err, [mf - v])
            elif k == "lr":
                err = np.append(err, [lrf - v])
            elif k == "ltheta":
                err = np.append(err, [lthf - v])
            elif k == "lphi":
                err = np.append(err, [lphf - v])
            elif k == "lu":
                err = np.append(err, [luf - v])
            elif k == "lv":
                err = np.append(err, [lvf - v])
            elif k == "lw":
                err = np.append(err, [lwf - v])            
            elif k == "lm":
                err = np.append(err, [lmf - v])
            else:
                raise ValueError(f"Unknown boundary condition: {k}")

        return err


@njit
def du_stable(r, v, w):
    inv_r = 1.0/np.float64(r)
    rv, rw = r*np.float64(v), r*np.float64(w)
    h2 = rv*rv + rw*rw
    du = (h2 - r) * (inv_r**3)   # = (v^2+w^2)/r - 1/r^2
    return du

@njit
def dv_stable(r, u, v, w, ph):
    inv_r = 1.0/np.float64(r)
    sphi, cphi = np.sin(ph), np.cos(ph)
    v_tan_phi = (np.float64(v) * sphi) / cphi
    dv = -(u*v)*inv_r + (w*inv_r)*v_tan_phi
    return dv

@njit
def dw_stable(r, u, v, w, ph):
    inv_r = 1.0/np.float64(r)
    sphi, cphi = np.sin(ph), np.cos(ph)
    v_over_r   = np.float64(v) * inv_r
    v_tan_phi  = (np.float64(v) * sphi) / cphi
    dw = -(u*w)*inv_r - v_over_r * v_tan_phi
    return dw

@njit
def swfun(fullstate, data):
    r, th, ph, u, v, w, m, lr, lth, lph, lu, lv, lw, lm = fullstate
    _, u_e, _, _ = data
    
    LV = np.sqrt(lu**2 + lv**2 + lw**2)
    return LV - lm*m/u_e, LV

@njit
def interpolate_rho_4d(t, r, theta, phi, rho_matrix):
    r *= 6371.0  # Dimenisonalization: r in km
    t *= (3600*24)  # Dimenisonalization: t in seconds DA MODIFICARE ANCORA
    theta = theta % (2*np.pi)
    t_vals = np.linspace(0, 3600*24*1, 5)
    r_vals = np.linspace(6600, 7000, 30)
    theta_vals = np.linspace(0, 2*np.pi, 37)
    phi_vals = np.linspace(-np.pi/2, np.pi/2, 19)

    def find_index(val, arr):
        for i in range(len(arr)-1):
            if arr[i] <= val <= arr[i+1]:
                return i
        return len(arr)-2

    it = find_index(t, t_vals)
    ir = find_index(r, r_vals)
    iphi = find_index(phi, phi_vals)
    ith = find_index(theta, theta_vals)

    t0, t1 = t_vals[it], t_vals[it+1]
    r0, r1 = r_vals[ir], r_vals[ir+1]
    phi0, phi1 = phi_vals[iphi], phi_vals[iphi+1]
    th0, th1 = theta_vals[ith], theta_vals[ith+1]

    dt = (t - t0) / (t1 - t0) if t1 != t0 else 0
    dr = (r - r0) / (r1 - r0) if r1 != r0 else 0
    dp = (phi - phi0) / (phi1 - phi0) if phi1 != phi0 else 0
    dth = (theta - th0) / (th1 - th0) if th1 != th0 else 0

    rho = 0.0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    weight = (1 - dt if a == 0 else dt) * \
                             (1 - dr if b == 0 else dr) * \
                             (1 - dp if c == 0 else dp) * \
                             (1 - dth if d == 0 else dth)
                    rho += weight * rho_matrix[it+a, ir+b, iphi+c, ith+d]
    return rho

@njit
def ode_orb3d(t, fullstate, data, TS=None):
    r, th, ph, u, v, w, m, lr, lth, lph, lu, lv, lw, lm = fullstate
    T_max, u_e, delta, rho_matrix = data
    rho_interp = interpolate_rho_4d(t, r, th, ph, rho_matrix)
    
    # switching function
    sf, LV = swfun(fullstate, data)
    
    T = compute_thrust(TS, sf, T_max)
    
    au, av, aw = 0.0, 0.0, 0.0
    
    a_drag_zen = compute_drag_acceleration_zen(r, th, ph, u, v, w, m, rho_interp)
    
    a_drag_u = a_drag_zen[0]
    a_drag_v = a_drag_zen[1]
    a_drag_w= a_drag_zen[2]
    
    # a_drag_u, a_drag_v, a_drag_w = 0.0, 0.0, 0.0
    
    if (T > 0.0) and (LV > 1e-12):
        au = (T/m) * (lu / LV) + a_drag_u
        av = (T/m) * (lv / LV) + a_drag_v
        aw = (T/m) * (lw / LV) + a_drag_w
    else:
        au = a_drag_u
        av = a_drag_v
        aw = a_drag_w
    
    # print(sf, T, au, av, aw)
    
    J = compute_total_derived_perturbations(r, th, ph, u, v, w, m, rho_interp)
    # # Estrai righe: una per ogni componente dell'accelerazione (u, v, w)
    da_u_dr, da_u_dth, da_u_dph, da_u_dm = J[0, :]
    da_v_dr, da_v_dth, da_v_dph, da_v_dm = J[1, :]
    da_w_dr, da_w_dth, da_w_dph, da_w_dm = J[2, :]
    
    # da_u_dr, da_u_dth, da_u_dph, da_u_dm = 0.0, 0.0, 0.0, 0.0
    # da_v_dr, da_v_dth, da_v_dph, da_v_dm = 0.0, 0.0, 0.0, 0.0
    # da_w_dr, da_w_dth, da_w_dph, da_w_dm = 0.0, 0.0, 0.0, 0.0
    
    dfdt = np.zeros_like(fullstate)    
    
    # States
    dfdt[0] = u
    dfdt[1] = v/(r*cos(ph))
    dfdt[2] = w/r
    dfdt[3] = du_stable(r, v, w) + au
    dfdt[4] = dv_stable(r, u, v, w, ph) + av
    dfdt[5] = dw_stable(r, u, v, w, ph) + aw
    dfdt[6] = -T / u_e

    # Costates
    dfdt[7] = (1/r**2) * (lth*v/cos(ph) + lph*w + lu*(-2*1/r + v**2 + w**2) + lv*(-u*v + v*w*tan(ph)) + lw*(-u*w - v**2*tan(ph))) - lu*da_u_dr - lv*da_v_dr - lw*da_w_dr
    dfdt[8] = - lu*da_u_dth - lv*da_v_dth - lw*da_w_dth
    dfdt[9] = 1/(r*cos(ph)**2)*(-lth*v*sin(ph) - lv*v*w + lw*v**2) - lu*da_u_dph - lv*da_v_dph - lw*da_w_dph
    dfdt[10] = (1/r) * (-lr*r + lv*v + lw*w)
    dfdt[11] = (1/r) * (-lth/cos(ph) - 2*lu*v + lv*(u - w*tan(ph)) + 2*lw*v*tan(ph))
    dfdt[12] = (1/r) * (-lph - 2*lu*w - lv*v*tan(ph) + lw*u)
    dfdt[13] = (T*LV) / m**2 - lu*da_u_dm - lv*da_v_dm -lw*da_w_dm

    return dfdt

@njit
def compute_thrust(TS, sf, T_max):
    if TS is None:
        T = T_max if sf > 0 else 0.0
    else:
        T = T_max if TS == 1 else 0.0
    return T

@njit
def compute_H_orb3d(t, fullstate, data, TS=None):
    r, th, ph, u, v, w, m, lr, lth, lph, lu, lv, lw, lm = fullstate
    T_max, u_e, _, rho_matrix = data
    rho_interp = interpolate_rho_4d(t, r, th, ph, rho_matrix)
    
    sf, LV = swfun(fullstate, data)
    
    T = compute_thrust(TS, sf, T_max)
        
    au, av, aw = 0.0, 0.0, 0.0
    
    # a_drag_u, a_drag_v, a_drag_w = 0.0, 0.0, 0.0
    
    a_drag_zen = compute_drag_acceleration_zen(r, th, ph, u, v, w, m, rho_interp)
    
    a_drag_u = a_drag_zen[0]
    a_drag_v = a_drag_zen[1]
    a_drag_w = a_drag_zen[2]
    
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
        lu*(-1/r**2 + v**2/r + w**2/r + au) + 
        lv*(-u*v/r + v*(w/r)*tan(ph) + av) + 
        lw*(-u*w/r - (v**2/r)*tan(ph) + aw)
        - lm*T/u_e
    )

def event_r_below_one(t, y):
    """
    Evento esempio: se raggio scende sotto 1, fermati.
    """
    r = y[0]
    return r - 1.05

event_r_below_one.terminal = True   # Se si verifica, fermati
event_r_below_one.direction = -1    # La direzione di verifica è da r > 1.05 a r < 1.05

def event_r_above(t,y):
    r = y[0]
    return r - 1.55
event_r_above.terminal = True
event_r_above.direction = 1


