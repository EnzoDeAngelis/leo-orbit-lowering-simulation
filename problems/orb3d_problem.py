import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
import yaml
import os

from problems.base_problem import BaseProblem
from OCP.functions import swfun, ode_orb3d, compute_H_orb3d

class OrbitalProblem(BaseProblem):
    def __init__(self, initial_state=[1.2, 0.0, 0.0, 0.0, np.sqrt(1/1.2), 0.0, 1],     # Lo stato iniziale viene letto da qui solo se non esiste il file prob.yaml
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

    def compute_H(self, state, data, thr=None):
        return compute_H_orb3d(state, data, thr)
    
    def compute_sf(self, state, data):
        return swfun(state, data)

    def boundary_error(self, param_guess, data, timefree):
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

            rf, thf, phf, uf, vf, wf, mf, lrf, lthf, lphf, luf, lvf, lwf, lmf = sol.y[:, -1]
        # if event above fired, we want to penalize it
        
            sf = np.sqrt(
                sol.y[10,:]**2 + sol.y[11,:]**2 + sol.y[12,:]**2) - \
                    sol.y[-1,:]*sol.y[6,:]/data[1]
            
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


