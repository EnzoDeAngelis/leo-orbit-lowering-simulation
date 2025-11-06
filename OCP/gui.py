import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QDoubleSpinBox,
    QPushButton,
    QSpinBox,
    QCheckBox,
    QSizePolicy,
    QGridLayout,
    QComboBox,
    QGroupBox,
    QInputDialog
)
from PyQt5.QtCore import Qt
from problems.orb3d_problem import OrbitalProblem
from problems.base_problem import BaseProblem
from functions import calcolare_rho_matrix_4d
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.integrate import solve_ivp
import json
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import yaml


R = 6371.0  # Earth radius in km
MU = 398600      # Earth gravitational constant in km^3/s^2
V = np.sqrt(MU / R)  # Circular orbit velocity in km/s
T = R / V        # dimensional time unit in s
A = V**2 / R # dimensional acceleration unit in km/s²
M = 500 # reference mass in kg
TN = M * A # reference thrust in kN

tols = 1e-6
int_method = "RK45"

###############################################################################
# 1) ODE GUI
###############################################################################

def _fd_column(i, base_err, pg, method, timefree, problem, data):
    pg = pg.copy()
    jj = float(data[2])
    if not timefree and i == len(pg)-1:
        return i, np.zeros_like(base_err)
    if method == "central":
        pg[i] += jj
        e_plus = problem.boundary_error(pg, data, timefree,tols=tols,method=int_method)
        pg[i] -= 2*jj
        e_minus = problem.boundary_error(pg, data, timefree,tols=tols,method=int_method)
        col = (e_plus - e_minus) / (2*jj)
    else:  # forward/backward
        step = +jj if method == "forward" else -jj
        pg[i] += step
        e_plus = problem.boundary_error(pg, data, timefree,tols=tols,method=int_method)
        col = (e_plus - base_err) / step
    return i, col

def compute_jacobian_fd(problem, param_guess, data, method="forward", timefree=True, max_workers= os.cpu_count() - 1):
    pg = np.asarray(param_guess, float)
    base_err = problem.boundary_error(pg, data, timefree, tols=tols, method=int_method)
    n_eff = len(pg) if timefree else len(pg)-1
    J = np.zeros((len(base_err), n_eff))
    iters = zip(range(n_eff),
                repeat(base_err), repeat(pg), repeat(method),
                repeat(timefree), repeat(problem), repeat(data))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, col in ex.map(lambda args: _fd_column(*args), iters):
            J[:, i] = col
    return J, base_err

def compute_corrections(J, err, reg=0.0, rcond=1e-12):
    J = np.asarray(J, dtype=np.float64)
    err = np.asarray(err, dtype=np.float64)

    # scaling colonne
    col_scale = np.maximum(np.linalg.norm(J, axis=0), 1.0)
    Js = J / col_scale

    JTJ = Js.T @ Js
    g   = Js.T @ err
    if reg > 0:
        JTJ = JTJ + reg * np.eye(JTJ.shape[0])

    d_scaled, *_ = np.linalg.lstsq(JTJ, -g, rcond=rcond)
    d = d_scaled / col_scale
    return d


def newton_shoot_step(problem, param_guess, data,
                      alpha=0.5, method="forward", timefree=False,
                      c=1e-4, tau_d=0.5, tau_u=1.5, eta_f=0.25, reg=0.0):

    pg = np.asarray(param_guess, dtype=np.float64)

    J, err0 = compute_jacobian_fd(problem, pg, data, method=method, timefree=timefree)
    dparam  = compute_corrections(J, err0, reg=reg)
    phi0    = 0.5 * float(err0 @ err0)

    def apply_step(pg_arr, a, dp):
        if timefree:
            if dp.shape[0] != len(pg_arr):
                raise ValueError(f"dparam has length {dp.shape[0]} but param_guess has {len(pg_arr)} (timefree=True).")
            return pg_arr + a * dp
        else:
            if dp.shape[0] != (len(pg_arr) - 1):
                raise ValueError(f"dparam has length {dp.shape[0]} but param_guess-1 is {len(pg_arr)-1} (timefree=False).")
            return np.append(pg_arr[:-1] + a * dp, pg_arr[-1])

    alpha_ls = float(alpha)
    trial = apply_step(pg, alpha_ls, dparam)

    # Backtracking (Armijo)
    for _ in range(10):
        _, err_trial = compute_jacobian_fd(problem, trial, data, method=method, timefree=timefree)
        phi_trial = 0.5 * float(err_trial @ err_trial)
        if phi_trial <= (1 - c * alpha_ls) * phi0:
            break
        alpha_ls *= tau_d
        trial = apply_step(pg, alpha_ls, dparam)
    else:
        # fallback: Newton steepest
        grad = J.T @ err0
        ngrad = np.linalg.norm(grad)
        if ngrad < 1e-16:
            # nessun miglioramento possibile
            return pg, err0, np.sqrt(2*phi0), alpha_ls
        d_sd = -grad / ngrad
        alpha_ls = alpha * 0.1
        trial = apply_step(pg, alpha_ls, d_sd)
        _, err_trial = compute_jacobian_fd(problem, trial, data, method=method, timefree=timefree)
        phi_trial = 0.5 * float(err_trial @ err_trial)

    # Forward-tracking
    alpha_next = min(alpha_ls * tau_u, 1.0) if (phi_trial < eta_f * phi0) else alpha_ls

    return trial, err_trial, np.sqrt(2*phi_trial), alpha_next


###############################################################################
# 3) Integration purely for plotting (costates + time separated)
###############################################################################
def integrate_with_sf(problem: BaseProblem, lam_guess, data, tfs=[20], n_points=300):
    """
    Integrates the system from t=0..t_final with the given costate guess lam_guess (5D).
    data = [T_max, u_e, delta, rho_matrix]

    Returns (time_array, states_array, sf_array).
    states_array has shape (10, len(time_array)).
    sf_array is the switching function array.
    """
    full0 = [*problem.initial_state, *lam_guess]

    if tfs[-1] <= 0:
        t_array = np.array([0])
        states = np.array(full0).reshape(-1, 1)
        sf_array = np.array([0])
        return t_array, states, sf_array

    NA = 1
    if problem.switching_structure is not None:
        NA = len(problem.switching_structure)
    t_start = 0
    all_t = []
    all_states = []
    sf_vals = []
    H_vals = []

    for arc_idx in range(NA):
        thr = None
        if problem.switching_structure is not None:
            thr = problem.switching_structure[arc_idx]
        t_end = tfs[arc_idx]
        sol = solve_ivp(
            lambda t, s: problem.ode_func(t, s, data, thr),
            (t_start, t_end),
            full0,
            dense_output=True,
            method=int_method,
            atol=tols,
            rtol=tols,
        )
        full0 = sol.y[:, -1]
        t_start = t_end
        all_t.append(sol.t)
        all_states.append(sol.y)

        # Evaluate switching function at each point
        for i in range(len(sol.t)):
            st = sol.y[:, i]
            sf_val, _ = problem.compute_sf(st, data)
            sf_vals.append(sf_val)
            H = problem.compute_H(sol.t[i], st, data, thr)
            H_vals.append(H)

    return all_t, all_states, np.array(sf_vals), np.array(H_vals)


###############################################################################
# 4) The PyQt GUI
###############################################################################
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        self.fig = fig


class PlanarGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCULUS v0.3.6")
        self.resize(1400, 900)

        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.main_layout = QVBoxLayout(self.widget)
        self.preverr = np.inf
        self.err = np.inf
        self.trueplot = True
        self.timefree = True
        self.problems = [OrbitalProblem()]
        self.current_problem = self.problems[0]
        self.plot_every = 100

        top_layout = QHBoxLayout()
        self.main_layout.addLayout(top_layout)

        # ---------- 1) CONTROLS ----------
        box_controls = QGroupBox("Controls")
        top_layout.addWidget(box_controls)

        grid_controls = QGridLayout(box_controls)
        grid_controls.setContentsMargins(3, 3, 3, 3)
        grid_controls.setHorizontalSpacing(4)
        grid_controls.setVerticalSpacing(2)

        row = 0
        grid_controls.addWidget(QLabel("Select problem:"), row, 0, Qt.AlignRight)
        self.combo_problem = QComboBox()
        for p in self.problems:
            self.combo_problem.addItem(p.name())
        self.combo_problem.currentIndexChanged.connect(self.on_problem_changed)
        grid_controls.addWidget(self.combo_problem, row, 1)
        row += 1

        grid_controls.addWidget(QLabel("alpha (relax):"), row, 0, Qt.AlignRight)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(1e-9, 1.0)
        self.spin_alpha.setDecimals(8)
        self.spin_alpha.setValue(1e-5)
        grid_controls.addWidget(self.spin_alpha, row, 1)

        self.chk_alpha_adapt = QCheckBox("Adattivo!")
        self.chk_alpha_adapt.setChecked(False)
        grid_controls.addWidget(self.chk_alpha_adapt, row, 2)
        row += 1

        grid_controls.addWidget(QLabel("delta (FD step):"), row, 0, Qt.AlignRight)
        self.spin_delta = QDoubleSpinBox()
        self.spin_delta.setRange(1e-9, 1e-2)
        self.spin_delta.setDecimals(7)
        self.spin_delta.setValue(1e-7)
        grid_controls.addWidget(self.spin_delta, row, 1)
        row += 1

        grid_controls.addWidget(QLabel("FD method:"), row, 0, Qt.AlignRight)
        self.combo_fdmethod = QComboBox()
        self.combo_fdmethod.addItems(["forward", "backward", "central"])
        self.combo_fdmethod.setCurrentIndex(0)
        grid_controls.addWidget(self.combo_fdmethod, row, 1)
        row += 1

        grid_controls.addWidget(QLabel("Step freccette: "), row, 0, Qt.AlignRight)
        self.spin_exp = QSpinBox()
        self.spin_exp.setRange(-6, 6)
        self.spin_exp.setValue(0)
        self.spin_exp.valueChanged.connect(self.on_step_changed)
        grid_controls.addWidget(self.spin_exp, row, 1)

        # ---------- 2) GUESS ----------
        box_guess_lam = QGroupBox("Guesses")
        top_layout.addWidget(box_guess_lam)
        self.grid_guess_lam = QGridLayout(box_guess_lam)

        self.costate_spins: list[QDoubleSpinBox] = []

        def refresh_costate_spins():
            while self.costate_spins:
                sp = self.costate_spins.pop()
                lbl = self.grid_guess_lam.itemAt(
                    self.grid_guess_lam.indexOf(sp) - 1
                ).widget()
                self.grid_guess_lam.removeWidget(sp)
                sp.deleteLater()
                self.grid_guess_lam.removeWidget(lbl)
                lbl.deleteLater()
            for i, name in enumerate(self.current_problem.costates()):
                lbl = QLabel(f"{name}0:")
                sp = QDoubleSpinBox()
                sp.setRange(-1e6, 1e6)
                sp.setDecimals(9)
                sp.setMaximumWidth(110)
                sp.valueChanged.connect(self.on_plot)
                row, col = divmod(i, 2)
                self.grid_guess_lam.addWidget(lbl, row, 2 * col)
                self.grid_guess_lam.addWidget(sp, row, 2 * col + 1)
                self.costate_spins.append(sp)

        refresh_costate_spins()  # prima creazione
        self.refresh_costate_spins = refresh_costate_spins  # salva per uso esterno

        # ---------- 3) GUESS tempi ----------
        box_guess_t = QGroupBox("Guess tempi")
        top_layout.addWidget(box_guess_t)
        self.grid_guess_t = QGridLayout(box_guess_t)

        self.time_guess_spins: list[QDoubleSpinBox] = []

        def refresh_time_guess_spins():
            # pulizia
            while self.time_guess_spins:
                sp = self.time_guess_spins.pop()
                lbl = self.grid_guess_t.itemAt(
                    self.grid_guess_t.indexOf(sp) - 1
                ).widget()
                self.grid_guess_t.removeWidget(sp)
                sp.deleteLater()
                self.grid_guess_t.removeWidget(lbl)
                lbl.deleteLater()
            # nuovo elenco
            NA = 1
            if self.current_problem.switching_structure is not None:
                NA = len(self.current_problem.switching_structure)
            for k in range(NA - 1):
                lbl = QLabel(f"t{k+1}:")
                sp = QDoubleSpinBox()
                row = k // 2
                col = k % 2
                self.grid_guess_t.addWidget(lbl, row, 2 * col)
                self.grid_guess_t.addWidget(sp, row, 2 * col + 1)
                self.time_guess_spins.append(sp)
            # tf
            lbl_tf = QLabel("tf:")
            sp_tf = QDoubleSpinBox()
            row = (NA - 1) // 2
            col = (NA - 1) % 2
            self.grid_guess_t.addWidget(lbl_tf, row, 2 * col)
            self.grid_guess_t.addWidget(sp_tf, row, 2 * col + 1)
            self.time_guess_spins.append(sp_tf)
            # set comuni
            for sp in self.time_guess_spins:
                sp.setRange(0.0, 1e6)
                sp.setDecimals(8)
                sp.setMaximumWidth(110)
                sp.valueChanged.connect(self.on_plot)

        refresh_time_guess_spins()
        self.refresh_time_guess_spins = refresh_time_guess_spins

        # checkbox tempo libero
        self.chk_timefree = QCheckBox("Final time free?")
        self.chk_timefree.setChecked(self.timefree)
        self.chk_timefree.toggled.connect(self.on_timefree_toggled)
        self.grid_guess_t.addWidget(
            self.chk_timefree, (len(self.time_guess_spins) + 1) // 2, 0, 1, 2
        )

        # ---------- 4) ACTIONS ----------
        box_actions = QGroupBox("Actions")
        top_layout.addWidget(box_actions)

        grid_actions = QGridLayout(box_actions)
        grid_actions.setContentsMargins(3, 3, 3, 3)
        grid_actions.setHorizontalSpacing(6)
        grid_actions.setVerticalSpacing(3)

        # riga 0 – Plot
        self.btn_plot = QPushButton("Plot")
        grid_actions.addWidget(self.btn_plot, 0, 0, 1, 2)
        self.btn_plot.clicked.connect(self.on_plot)

        # riga 1 – Shoot single | multi
        self.btn_shoot_single = QPushButton("Shoot - single")
        self.btn_shoot_multi = QPushButton("Shoot - multi")
        grid_actions.addWidget(self.btn_shoot_single, 1, 0)
        grid_actions.addWidget(self.btn_shoot_multi, 1, 1)
        self.btn_shoot_single.clicked.connect(self.on_shoot_single)
        self.btn_shoot_multi.clicked.connect(self.on_shoot_multi)

        # riga 2 – Steps spinner
        grid_actions.addWidget(QLabel("Steps:"), 2, 0, Qt.AlignRight)
        self.spin_multistep = QSpinBox()
        self.spin_multistep.setRange(1, 5000)
        self.spin_multistep.setValue(10)
        grid_actions.addWidget(self.spin_multistep, 2, 1)

        # riga 3 – Reset | Save
        # self.btn_reset = QPushButton("Reset")
        # self.btn_save = QPushButton("Save JSON")
        # grid_actions.addWidget(self.btn_reset, 3, 0)
        # grid_actions.addWidget(self.btn_save, 3, 1)
        # self.btn_reset.clicked.connect(self.on_reset)
        # self.btn_save.clicked.connect(self.on_save_json)
        
        self.btn_reset = QPushButton("Reset")
        self.btn_save_JSON = QPushButton("Save JSON")
        self.btn_save_yaml = QPushButton("Save yaml")
        grid_actions.addWidget(self.btn_reset, 3, 1)
        grid_actions.addWidget(self.btn_save_JSON, 4, 0)
        grid_actions.addWidget(self.btn_save_yaml, 4, 1)        
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save_JSON.clicked.connect(self.on_save_json)
        self.btn_save_yaml.clicked.connect(self.on_save_yaml)

        # -------------------------------------------------------------
        # (C) Third row: updated param (read-only for costates + final time?)
        self.third_layout = QGridLayout()
        self.main_layout.addLayout(self.third_layout)

        # λ-update (riga 0, sempre 7 elementi)
        lambda_labels = ["lr", "lth", "lph", "lu", "lv", "lw", "lm"]
        self.upd_lambda_spins = []
        for col, lab in enumerate(lambda_labels):
            lbl = QLabel(f"{lab}_upd:")
            sp = QDoubleSpinBox()
            self.third_layout.addWidget(lbl, 0, 2 * col)
            self.third_layout.addWidget(sp, 0, 2 * col + 1)
            self.upd_lambda_spins.append(sp)

        # Contenitore che verrà popolato/ripopolato a runtime
        self.time_spinboxes: list[QDoubleSpinBox] = []

        # prima creazione
        self.refresh_time_spinboxes()

        # -------------------------------------------------------------
        # (D) The embedded Matplotlib figure
        # -------------------------------------------------------------
        self.canvas = MplCanvas(self)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.main_layout.addWidget(self.canvas)

        self.lbl_err = QLabel("")
        self.lbl_err.setAlignment(Qt.AlignCenter)
        self.lbl_err.setWordWrap(True)
        self.main_layout.addWidget(self.lbl_err)

        self.axes = self.canvas.fig.subplots(4, 4).flatten()
        for ax in self.axes:
            ax.grid(True)

        # Rocket parameters
        self.T_max = 0.025 / (TN * 10**3)
        self.u_e = 20000.0 / (V * 10**3)
        epoch = datetime.datetime(2025, 3, 21, 12)
        self.rho_matrix = calcolare_rho_matrix_4d(epoch)
        self.lam_initial = None  # store initial param guess for "Reset"
        self.newton_iter = 0
        self.update_spin_steps()

    ####################################################################
    #   Utility: read/write param guess
    ####################################################################
    def refresh_time_spinboxes(self):
        """
        Ricrea le spinbox dei tempi di switch + tf in base alla lunghezza
        di self.current_problem.switching_structure (NA archi = NA−1 switch).
        Va chiamata quando cambia il problema o la switching function.
        """
        while self.time_spinboxes:
            sp = self.time_spinboxes.pop()
            lbl = self.third_layout.itemAt(self.third_layout.indexOf(sp) - 1).widget()
            self.third_layout.removeWidget(sp)
            sp.deleteLater()
            self.third_layout.removeWidget(lbl)
            lbl.deleteLater()

        NA = 1
        if self.current_problem.switching_structure is not None:
            NA = len(self.current_problem.switching_structure)

        for k in range(NA - 1):
            lbl = QLabel(f"t{k+1}_upd:")
            sp = QDoubleSpinBox()
            self.third_layout.addWidget(lbl, 1, 2 * k)
            self.third_layout.addWidget(sp, 1, 2 * k + 1)
            self.time_spinboxes.append(sp)

        lbl_tf = QLabel("tf_upd:")
        sp_tf = QDoubleSpinBox()
        self.third_layout.addWidget(lbl_tf, 1, 2 * (NA - 1))
        self.third_layout.addWidget(sp_tf, 1, 2 * (NA - 1) + 1)
        self.time_spinboxes.append(sp_tf)

        self.updated_spins = self.upd_lambda_spins + self.time_spinboxes
        for sp in self.updated_spins:
            sp.setRange(-1e8, 1e8)
            sp.setDecimals(9)
            sp.setMaximumWidth(100)

    def on_timefree_toggled(self, checked):
        """
        Called whenever the user checks or unchecks the "Time is Free?" box.
        `checked` is True if the box is checked, otherwise False.
        """
        self.timefree = checked

    def _time_param_names(self):
        NA = 1
        if self.current_problem.switching_structure is not None:
            NA = len(self.current_problem.switching_structure)
        return [f"t{i+1}" for i in range(NA - 1)] + ["tf"]

    def _costate_names(self):
        return list(self.current_problem.costates())

    def on_save_json(self):
        
        comment, ok = QInputDialog.getText(None, "Salva JSON", "Inserisci una breve descrizione:")
        if not ok:
            print("Salvataggio annullato.")
            return
        
        names_cost = list(self.current_problem.costates())
        names_time = self._time_param_names()
        full_names = names_cost + names_time

        entry = {
            "comment": comment,
            "T_max": self.T_max,
            "u_e": self.u_e,
            "alpha": self.spin_alpha.value(),
            "delta": self.spin_delta.value(),
            "fd_method": self.combo_fdmethod.currentText(),
            "newton_iter": self.newton_iter,
        }

        if self.lam_initial is not None:
            entry["lam_initial"] = {n: v for n, v in zip(full_names, self.lam_initial)}

        lam_upd = self.read_updated_param()
        entry["lam_updated"] = {n: v for n, v in zip(full_names, lam_upd)}

        all_data = {}
        if os.path.exists("output.json"):
            with open("output.json", "r") as fp:
                try:
                    all_data = json.load(fp)
                except json.JSONDecodeError:
                    print("File corrotto, skip del salvataggio. VERIFICA il file.")
                    return

        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        all_data[timestamp] = entry

        with open("output.json", "w") as fp:
            json.dump(all_data, fp, indent=2)

        print(f"Saved current data to 'output.json' under key {timestamp}.")
        
    def on_save_yaml(self):

        names = ["r", "theta", "phi", "u", "v", "w", "m"]
        names_cost = list(self.current_problem.costates())
        names_time = self._time_param_names()
        full_names = names_cost + names_time

        lam_upd = self.read_updated_param()
        delta = self.spin_delta.value()
        data = (self.T_max, self.u_e, delta, self.rho_matrix)


        NA = len(self._time_param_names())
        guesses = lam_upd[:7]
        tfs = lam_upd[7:7 + NA]

        t_array, states, sf_array, H_array = integrate_with_sf(
            self.current_problem, guesses, data, tfs)
    
        full_states = np.hstack(states)
        r, theta, phi, u, v, w, m = full_states[:7, -1]
        param_upd = [r, theta, phi, u, v, w, m]

        param_dict = {k: float(v) for k, v in zip(names, param_upd)}

        yaml_data = {
            "param_updated": param_dict
        }

        updated_only = {
            "costates": {n: float(v) for n, v in zip(names_cost, lam_upd[:7])},
            "times": {n: float(v) for n, v in zip(names_time, lam_upd[7:])},
        }

        with open("output.yaml", "w") as f:
            yaml.dump(yaml_data, f, sort_keys=False)

        with open(f"updated_values.yaml", "w") as f:
            yaml.dump(updated_only, f, sort_keys=False)

        print(f"Salvati parametri aggiornati.")

        # Crea la cartella se non esiste
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)  # Ignora se già esistente
        # Estrai tutti i valori da salvare (costates + times)
        all_vars = {
            **{n: float(v) for n, v in zip(names_cost, lam_upd[:7])},  # Costates
            **{n: float(v) for n, v in zip(names_time, lam_upd[7:])}   # Times
        }

        # Salvataggio unico dei tempi su una riga
        time_values = [float(v) for v in lam_upd[7:]]  # Prendi solo i tempi aggiornati
        time_line = " ".join(f"{v:.9f}" for v in time_values)  # Format a 9 decimali

        filepath = os.path.join(output_dir, "tempi.txt")
        with open(filepath, "a") as f:
            f.write(time_line + "\n")  # Scrive tutti i tempi su una riga


        for var_name, value in param_dict.items():
            filepath = os.path.join(output_dir, f"{var_name}.txt")
            with open(filepath, "a") as f:
                f.write(f"{value}\n")

        print(f"Valori aggiunti ai file separati in '{output_dir}/'.")

        # 1. Recupero errori dall'ultima iterazione
        if hasattr(self, 'last_err'):
            err_final = self.last_err
        else:
            # Se last_err non esiste, usiamo l'output di _update_error_label
            err_final = np.zeros(10)  # Array di fallback
            
        # 2. Nomi errori come da tua richiesta
        error_names = ["SF_1", "SF_2", "H", "r", "theta", "phi", "u", "v", "w", "m"]
        
        # 3. Salvataggio nello stesso stile delle altre variabili
        for err_name, err_value in zip(error_names, err_final[:len(error_names)]):
            filepath = os.path.join(output_dir, f"err_{err_name}.txt")
            with open(filepath, "a") as f:
                f.write(f"{float(err_value):.8e}\n")

        # 4. Metriche aggiuntive
        max_err = max(abs(err_final)) if len(err_final) > 0 else 0.0
        norm_err = np.linalg.norm(err_final) if len(err_final) > 0 else 0.0
        
        with open(os.path.join(output_dir, "err_max.txt"), "a") as f:
            f.write(f"{max_err:.8e}\n")
        
        with open(os.path.join(output_dir, "err_norm.txt"), "a") as f:
            f.write(f"{norm_err:.8e}\n")

        print(f"Errori salvati in stile coerente in '{output_dir}/'")

    def read_param_guess(self):
        """
        Concatena costate-guess + tempi-guess in un'unica lista.
        """
        vals = [sp.value() for sp in self.costate_spins]
        vals.extend(sp.value() for sp in self.time_guess_spins)
        return vals

    def write_updated_param(self, param):
        n = len(self.upd_lambda_spins)
        for i, sp in enumerate(self.upd_lambda_spins):
            sp.setValue(param[i])
        for j, sp in enumerate(self.time_spinboxes):
            sp.setValue(param[n + j])

    ####################################################################
    #   Spin steps
    ####################################################################
    def on_step_changed(self):
        self.update_spin_steps()

    def update_spin_steps(self):
        exponent = self.spin_exp.value()
        step_size = 10.0**exponent
        for sp in self.costate_spins:
            sp.setSingleStep(step_size)

    ####################################################################
    #   Plotting the trajectory
    ####################################################################
    def plot_colored_segments(
        self, ax, t, y, sf, default_color="C0", highlight_color="orange"
    ):
        indices = np.where(np.diff(np.sign(sf)))[0]  # Find sign-change indices
        segments = np.split(
            np.arange(len(t)), indices + 1
        )  # Split indices into segments

        for seg in segments:
            color = highlight_color if sf[seg[0]] > 0 else default_color
            ax.plot(t[seg], y[seg], color=color)

    def plot_trajectory(self, guesses, data, tfs):
        if not self.trueplot:
            return

        t_array, states, sf_array, H_array = integrate_with_sf(
            self.current_problem, guesses, data, tfs
        )

        # Make a single array
        t_array = np.concatenate(t_array)
        states = np.concatenate(states, axis=1)

        for ax in self.axes:
            ax.clear()
            ax.grid(True)

        labels = (
            self.current_problem.state_labels()
            + self.current_problem.costate_labels()
            + ["SF(t)", "H(t)"]
        )

        for i in range(len(labels) - 2):
            lab = labels[i]
            ax = self.axes[i]
            self.plot_colored_segments(ax, t_array, states[i, :], sf_array)
            ax.set_title(rf"${lab}$")

        # SF(t)
        self.plot_colored_segments(self.axes[14], t_array, sf_array, sf_array)
        self.axes[14].set_title("SF(t)")

        # Hamiltonian
        self.plot_colored_segments(self.axes[15], t_array, H_array, sf_array)
        self.axes[15].set_title("Hamiltonian(t)")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    ####################################################################
    #   Newton iteration
    ####################################################################
    def _update_error_label(self, err_vec):
        NA = 1
        names_sf = []
        if self.current_problem.switching_structure is not None:
            NA = len(self.current_problem.switching_structure)
            names_sf = [f"Sf{i+1}" for i in range(NA - 1)] or ["Sf"]
        if self.timefree:
            names_sf.append("Hf")
        names_bc = list(self.current_problem.boundary_conditions.keys())
        names_all = names_sf + names_bc

        def colour(val):
            a = abs(val)
            if a > 1e-3:
                return "#ff0000"
            if a > 1e-5:
                return "#ff8c00"
            if a > 1e-6:
                return "#0000ff"
            return "#228b22"

        chunks = [
            f'<span style="color:{colour(e)}">{n}={e:+.2e}</span>'
            for n, e in zip(names_all, err_vec)
        ]
        norm_err = np.linalg.norm(err_vec)
        max_err = max(abs(err_vec))
        chunks.append(
            f'<span style="color:{colour(norm_err)}">‖err‖={norm_err:.2e}</span>'
        )
        chunks.append(f'<span style="color:{colour(max_err)}">max={max_err:.2e}</span>')
        html = "  |  ".join(chunks)
        self.lbl_err.setText(f"<b>{html}</b>")

    def do_one_newton_iteration(self, guesses):
        alpha_curr = self.spin_alpha.value()
        delta = self.spin_delta.value()
        data = (self.T_max, self.u_e, delta, self.rho_matrix)

        fd_method = self.combo_fdmethod.currentText()
        adaptive = self.chk_alpha_adapt.isChecked()
        timefree = self.timefree

        param_next, err, norm_err, alpha_new = newton_shoot_step(
            self.current_problem,
            guesses,
            data,
            alpha=alpha_curr,
            method=fd_method,
            timefree=timefree
        )

        if adaptive:
            self.spin_alpha.setValue(alpha_new)
        self.preverr = norm_err
        self.newton_iter += 1
        print(f"Iter {self.newton_iter:4d}")
        print("Guess: ", end="")
        for i, p in enumerate(guesses):
            print(f"{p: .4e} ", end="")
        print("\nNew g: ", end="")
        for i, p in enumerate(param_next):
            print(f"{p: .4e} ", end="")
        print(
            f"({np.linalg.norm(err): .4e}, {max(abs(err)):.4e})\nErr:   ", end=""
        )
        for i, e in enumerate(
            err
        ):
            print(f"{e: .4e} ", end="")
        print()

        self._update_error_label(err)
        self.write_updated_param(param_next)

        NA = 1
        if self.current_problem.switching_structure is not None:
            NA = len(self.current_problem.switching_structure)
        lambdas = param_next[:-NA]
        tfs = param_next[-NA:]
        self.plot_trajectory(lambdas, data, tfs)

        return param_next, err, norm_err

    ####################################################################
    #   Button Callbacks
    ####################################################################
    def on_problem_changed(self, idx):
        self.current_problem = self.problems[idx]
        self.refresh_costate_spins()
        self.refresh_time_guess_spins()

    def on_plot(self):
        sender = self.sender()
        if sender is None:
            return

        if isinstance(sender, QDoubleSpinBox):
            if (
                sender.hasFocus()
            ):  # Skip plotting if the spinbox has focus (manual typing)
                return
        param_guess = self.read_param_guess()
        if self.lam_initial is None:
            # store it
            self.lam_initial = param_guess.copy()

        alpha = self.spin_alpha.value()
        delta = self.spin_delta.value()
        data = (self.T_max, self.u_e, delta, self.rho_matrix)

        NA = 1
        if self.current_problem.switching_structure is not None:
            NA = len(self.current_problem.switching_structure)
        lambdas = param_guess[:-NA]
        ts = param_guess[-NA:]
        self.plot_trajectory(lambdas, data, ts)

    def read_updated_param(self):
        vals = [sp.value() for sp in self.upd_lambda_spins]
        vals.extend(sp.value() for sp in self.time_spinboxes)
        return vals

    def on_shoot_single(self):
        param_current = self.read_param_guess()
        if any(sp.value() != 0.0 for sp in self.updated_spins):
            param_current = self.read_updated_param()
        self.trueplot = True
        self.do_one_newton_iteration(param_current)  # always plot single step

    def on_shoot_multi(self):
        param_current = self.read_param_guess()
        if any(sp.value() != 0.0 for sp in self.updated_spins):
            param_current = self.read_updated_param()
        max_iter = self.spin_multistep.value()  # read from the new spin box
        for k in range(max_iter):
            self.trueplot = k % self.plot_every == 0 or k == max_iter - 1
            param_next, err, norm_err = self.do_one_newton_iteration(param_current)
            param_current = param_next
            QApplication.processEvents()
            if norm_err < 1e-6:
                print("Converged!")
                break

    def on_reset(self):
        """
        Riporta i parametri di guess allo stato salvato in lam_initial
        e azzera gli updated-values; newton_iter torna a 0.
        """
        self.trueplot = True

        if self.lam_initial is not None:
            nλ = len(self.costate_spins)

            # --- ripristina costati ---
            for i, sp in enumerate(self.costate_spins):
                sp.setValue(self.lam_initial[i])

            # --- ripristina tempi ---
            for j, sp in enumerate(self.time_guess_spins):
                sp.setValue(self.lam_initial[nλ + j])

        # azzera updated-param
        for sp in self.updated_spins:
            sp.setValue(0.0)

        self.newton_iter = 0
        print("Reset done: guess restored, updated param = 0")


def main():
    app = QApplication(sys.argv)
    gui = PlanarGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
