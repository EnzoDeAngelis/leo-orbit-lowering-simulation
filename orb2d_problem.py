import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
import yaml
import os

from problems.base_problem import BaseProblem

class OrbitalProblem(BaseProblem):
    def __init__(self, initial_state=[1.2, 0.0, 0.0, np.sqrt(1/1.2), 1],
                 switching_structure=[1, 0, 1]):
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
        return [-0.352952, -9.3e-05, -0.002506, -0.790984, 1.0, 3.500311]  # esempio default guess
    
    ##### DEVE ESSERE MODIFICATO DA PARTE VOSTRA IN BASE AL PROBLEMA #####
    def states(self):
        return ["r", "theta", "u", "v", "m"]
    
    ##### DEVE ESSERE MODIFICATO DA PARTE VOSTRA IN BASE AL PROBLEMA #####
    def costates(self):
        return ["lr", "ltheta", "lu", "lv", "lm"] 
    
    def state_labels(self):
        return [x + "(t)" for x in self.states()]
    
    def costate_labels(self):
        return ["\lambda " + x + "(t)" for x in self.states()]

    def ode_func(self, t, state, data, thr=None):
        return ode_orb2d(t, state, data, thr)

    def compute_H(self, state, data, thr=None):
        return compute_H_orb2d(state, data, thr)
    
    def compute_sf(self, state, data):
        return swfun(state, data)

    def boundary_error(self, param_guess, data, timefree):
        lr0, lth0, lu0, lv0, lm0, *tfs = param_guess

        full0 = [*self.initial_state, lr0, lth0, lu0, lv0, lm0]

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
                            t_eval=np.linspace(tstart, t_end, 100),
                            method="RK45",
                            events=[
                                event_r_below_one,
                                event_r_above]
                                )
            # if unsuccessful, return a large penalty
            # if sol.status != 0:
            #     return np.array([1e6])

            # if len(sol.t_events) > 0 and (
            #     len(sol.t_events[0]) > 0 or len(sol.t_events[1]) > 0):
            #     # The event fired => r dropped below 1 => INVALID TRAJECTORY
            #     # Return a LARGE penalty so the optimizer avoids this solution.
            #     return np.array([1e6])

            rf, thf, uf, vf, mf, lrf, lthf, luf, lvf, lmf = sol.y[:, -1]
        # if event above fired, we want to penalize it
        
            sf = np.sqrt(
                sol.y[7,:]**2 + sol.y[8,:]**2) / sol.y[4,:] - \
                    sol.y[-1,:]/data[1]
            
            # Se il problema è a tempo libero e siamo all'ultimo arco,
            # calcoliamo l'Hamiltoniana finale
            if timefree and arc_idx == NA-1:
                Hf = self.compute_H(sol.y[:, -1], data, thr)
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
                err = np.append(err, [rf - v])
            elif k == "theta":
                err = np.append(err, [thf - v])
            elif k == "u":
                err = np.append(err, [uf - v])
            elif k == "v":
                err = np.append(err, [vf - v])
            elif k == "m":
                err = np.append(err, [mf - v])
            elif k == "lr":
                err = np.append(err, [lrf - v])
            elif k == "ltheta":
                err = np.append(err, [lthf - v])
            elif k == "lu":
                err = np.append(err, [luf - v])
            elif k == "lv":
                err = np.append(err, [lvf - v])
            elif k == "lm":
                err = np.append(err, [lmf - v])
            else:
                raise ValueError(f"Unknown boundary condition: {k}")

        return err

@njit
def swfun(fullstate, data):
    r, th, u, v, m, lr, lth, lu, lv, lm = fullstate
    T_max, u_e, _ = data

    LV = np.sqrt(lu**2 + lv**2)
    return LV/m - lm/u_e, LV

@njit
def ode_orb2d(t, fullstate, data, TS=None):
    r, th, u, v, m, lr, lth, lu, lv, lm = fullstate
    T_max, u_e, _ = data

    # switching function
    sf, LV = swfun(fullstate, data)
    if TS is None:
        T = T_max if sf > 0 else 0.0
    else:
        T = T_max if TS == 1 else 0.0

    au, av = 0.0, 0.0
    if (T > 0.0) and (LV > 1e-12):
        au = (T/m) * (lu / LV)
        av = (T/m) * (lv / LV)
    
    dfdt = np.zeros_like(fullstate)

    # States
    dfdt[0] = u
    dfdt[1] = v/r
    dfdt[2] = -1/r**2 + v**2/r + au
    dfdt[3] = -u*v/r + av
    dfdt[4] = -T / u_e

    # Costates
    dfdt[5] = (1/r**2) * (lth*v -lu*(2/r - v**2) - lv*u*v)
    dfdt[6] = 0
    dfdt[7] = -lr + lv*v/r
    dfdt[8] = (1/r) * (-lth -2*lu*v + lv*u)
    dfdt[9] = (T*LV) / m**2

    return dfdt

@njit
def compute_H_orb2d(fullstate, data, TS=None):
    r, th, u, v, m, lr, lth, lu, lv, lm = fullstate
    T_max, u_e, _ = data

    sf, _ = swfun(fullstate, data)

    if TS is None:
        T = T_max if sf > 0 else 0.0
    else:
        T = T_max if TS == 1 else 0.0
    
    return (
        lr*u + lth*v/r +lu*(-1/r**2+v**2/r)
        - lv*u*v/r + T*sf
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


