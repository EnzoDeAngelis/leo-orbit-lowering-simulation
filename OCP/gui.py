import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

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
)
from PyQt5.QtCore import Qt
from problems.orb3d_problem import OrbitalProblem
from problems.base_problem import BaseProblem

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.integrate import solve_ivp
import json

R = 3396
V = 3.551255605022341
T = R / V
A = V**2 / R
M = 1000
TN = M * A

###############################################################################
# 1) ODE GUI
###############################################################################


def compute_jacobian_fd(
    problem: BaseProblem, param_guess, data, method="forward", timefree=True
):
    delta = data[2]
    base_err = problem.boundary_error(param_guess, data, timefree)
    n = len(param_guess)
    m = len(base_err)
    if not timefree:
        n -= 1  # exclude the final time from the Jacobian
    J = np.zeros((m, n))
    NA = 1
    if problem.switching_structure is not None:
        NA = len(problem.switching_structure)

    for i in range(n):
        pert = param_guess.copy()
        jj = delta

        if i == n - 1 and timefree:
            jj *= 100.0

        if method == "forward" or method == "backward":
            # forward difference
            pert[i] += jj if method == "forward" else -jj
            err_pert = problem.boundary_error(pert, data, timefree)
            J[:, i] = (err_pert - base_err) / jj
        else:
            # "central" difference
            # we do f(x+delta) - f(x-delta) all over 2*delta
            pert[i] += jj
            err_pert_plus = problem.boundary_error(pert, data, timefree)
            pert[i] -= 2 * jj
            err_pert_minus = problem.boundary_error(pert, data, timefree)
            J[:, i] = (err_pert_plus - err_pert_minus) / (2 * jj)

    return J, base_err


def compute_corrections(jacobian, errors):
    try:
        corr = np.linalg.solve(jacobian, -errors)
    except np.linalg.LinAlgError:
        corr = np.linalg.lstsq(jacobian, -errors, rcond=None)[0]
    return corr


def newton_shoot_step(
    problem: BaseProblem,
    param_guess,
    data,
    alpha=0.1,
    preverr=np.inf,
    method="forward",
    timefree=False,
    adaptive=False,
    c=1e-4,
    tau_d=0.9,  # back-tracking <1
    tau_u=1.001,  # forward-tracking >1
    eta_f=1,
):
    alpha_ls = alpha
    J, err0 = compute_jacobian_fd(
        problem, param_guess, data, method=method, timefree=timefree
    )
    dparam = compute_corrections(J, err0)
    norm0 = np.linalg.norm(err0)

    for _ in range(12):
        if timefree:
            trial = param_guess + alpha_ls * dparam
        else:
            trial = np.append(param_guess[:-1] + alpha_ls * dparam, param_guess[-1])

        _, err_trial = compute_jacobian_fd(
            problem, trial, data, method=method, timefree=timefree
        )
        norm_trial = np.linalg.norm(err_trial)

        if norm_trial <= (1 - c * alpha_ls) * norm0:
            break
        alpha_ls *= tau_d  # riduci passo e riprova (tipo bisezione di prima)
    else:
        # non soddisfa - teoricamente non va bene per niente se entri qui
        return param_guess, err0, norm0, max(alpha * tau_d, 1e-6)

    # ---------- forward-tracking (accelera) ----------
    alpha_next = alpha_ls
    if norm_trial < eta_f * norm0:  # miglioramento
        alpha_next = min(alpha_ls * tau_u, 1.0)  # prova passo +largo

    return trial, err_trial, norm_trial, alpha_next


###############################################################################
# 3) Integration purely for plotting (costates + time separated)
###############################################################################
def integrate_with_sf(problem: BaseProblem, lam_guess, data, tfs=[20], n_points=300):
    """
    Integrates the system from t=0..t_final with the given costate guess lam_guess (5D).
    data = [T_max, u_e, delta]

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
        if problem.switching_structure is not None:
            thr = problem.switching_structure[arc_idx]
        t_end = tfs[arc_idx]
        sol = solve_ivp(
            lambda t, s: problem.ode_func(t, s, data, thr),
            (t_start, t_end),
            full0,
            t_eval=np.linspace(t_start, t_end, n_points),
            dense_output=False,
            method="LSODA",
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
            H = problem.compute_H(st, data, thr)
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
        self.setWindowTitle("OCULUS v0.1")
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

        # -------------------------------------------------------------
        # (A) TOP ROW  – 4 group-boxes: Controls | Guess-λ | Guess-t | Actions
        # -------------------------------------------------------------
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
        self.btn_reset = QPushButton("Reset")
        self.btn_save = QPushButton("Save JSON")
        grid_actions.addWidget(self.btn_reset, 3, 0)
        grid_actions.addWidget(self.btn_save, 3, 1)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_save.clicked.connect(self.on_save_json)

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
        self.T_max = 0.05 / (TN * 10**3)
        self.u_e = 40000.0 / (V * 10**3)
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
        import datetime, os

        names_cost = list(self.current_problem.costates())
        names_time = self._time_param_names()
        full_names = names_cost + names_time

        entry = {
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

        # Loop through states and costates with proper segment coloring
        for i in range(len(labels) - 2):
            self.plot_colored_segments(self.axes[i], t_array, states[i, :], sf_array)
            self.axes[i].set_title(rf"${labels[i]}$")

        # SF(t) plot
        self.plot_colored_segments(
            self.axes[14], t_array, sf_array, sf_array
        )  # Cabio l'indice di self.axes (da [10]/[11] a [14]/[15]) sia per SF
        self.axes[14].set_title(
            "SF(t)"
        )  # che H in modo da avere la label giusta sui plot 3D

        # Hamiltonian plot
        self.plot_colored_segments(self.axes[15], t_array, H_array, sf_array)
        self.axes[15].set_title("Hamiltonian(t)")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    ####################################################################
    #   Newton iteration
    ####################################################################
    def _update_error_label(self, err_vec):
        NA = 1
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
        data = [self.T_max, self.u_e, delta]

        fd_method = self.combo_fdmethod.currentText()
        adaptive = self.chk_alpha_adapt.isChecked()
        # One iteration
        param_next, err, norm_err, alpha_new = newton_shoot_step(
            self.current_problem,
            guesses,
            data,
            alpha=alpha_curr,
            preverr=self.preverr,
            method=fd_method,
            timefree=self.timefree,
            adaptive=adaptive,
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
        )  # aggiunto il valore assoluto (abs(err)) in modo che stampi
        for i, e in enumerate(
            err
        ):  # il modulo del max errore residuo e non il massimo positivo
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
        data = [self.T_max, self.u_e, delta]

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
