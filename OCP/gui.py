import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QPushButton, QSpinBox, QCheckBox
)
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt

from problems.orb3d_problem import OrbitalProblem
from problems.base_problem import BaseProblem

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy.integrate import solve_ivp


R = 6371                     # Km
V = 7.909788019132536        # km/s
T = R/V
A = V**2/R
M = 500                      # Kg
TN = M * A

###############################################################################
# 1) ODE GUI
###############################################################################

def compute_jacobian_fd(
        problem: BaseProblem, param_guess, data, method="forward", timefree=True):
    delta = data[2]
    err, base_err = problem.boundary_error(param_guess, data, timefree)
    n = len(param_guess)
    m = len(base_err)
    if not timefree:
        n -= 1              # exclude the final time from the Jacobian
    J = np.zeros((m, n))
    NA = 1
    if problem.switching_structure is not None:
        NA = len(problem.switching_structure) 

    for i in range(n):
        pert = param_guess.copy()
        jj = delta

        if i == n-1 and timefree:
            jj *= 100.0  

        if method == "forward" or method == "backward":
            # forward difference
            pert[i] += jj if method == "forward" else -jj
            _, err_pert = problem.boundary_error(pert, data, timefree)
            J[:, i] = (err_pert - base_err) / jj
        else:
            # "central" difference
            # we do f(x+delta) - f(x-delta) all over 2*delta
            pert[i] += jj
            _, err_pert_plus = problem.boundary_error(pert, data, timefree)
            pert[i] -= 2*jj
            _, err_pert_minus = problem.boundary_error(pert, data, timefree)
            J[:, i] = (err_pert_plus - err_pert_minus) / (2 * jj)

    return J, base_err, err         # Faccio restituire anche err (all'interno del quale Hf non ha moltiplicatori). In questo modo posso chiamare
                                    # compute_jacobian_fd anche in newton_shoot_step
def compute_corrections(jacobian, errors):
    try:
        corr = np.linalg.solve(jacobian, -errors)
    except np.linalg.LinAlgError:
        corr = np.linalg.lstsq(jacobian, -errors, rcond=None)[0]
    return corr

def convergence_fun(err, norm_err):
    if max(abs(e) for e in err) < 1e-6:
        return True
    elif norm_err < 1e-4:
        return True
    else:
        return False
    
def newton_shoot_step(problem: BaseProblem, param_guess, data, alpha=0.1, preverr=np.inf,
                      method="forward", timefree=False):
    counter = 0
    valid = True
    #  while counter less than 10 and err is not less than preverr
    # halve alpha, recompute boundary_error_planar, and check again
    err = np.inf
    while counter < 10 and np.linalg.norm(err) > preverr and valid:
        J, _, err = compute_jacobian_fd(
            problem, param_guess, data, method=method, timefree=timefree)
        dparam = compute_corrections(J, err)
        if timefree:
            param_next = param_guess + alpha*dparam
        else:
            param_next = param_guess[:-1] + alpha*dparam
            param_next = np.append(param_next, param_guess[-1])
            err = np.append(err, 0)
        
        counter += 1
        alpha = alpha/2
        param_guess = param_next.copy()
    else:
        J, _, err = compute_jacobian_fd(
            problem, param_guess, data, method=method, timefree=timefree)
        dparam = compute_corrections(J, err)
        if timefree:
            param_next = param_guess + alpha*dparam
        else:
            param_next = param_guess[:-1] + alpha*dparam
            param_next = np.append(param_next, param_guess[-1])
            err = np.append(err, 0)
    
    norm_err = np.linalg.norm(err)
    return param_next, err, norm_err

###############################################################################
# 3) Integration purely for plotting (costates + time separated)
###############################################################################
def integrate_with_sf(problem : BaseProblem, lam_guess, data, tfs=[20], n_points=300):
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
        states = np.array(full0).reshape(-1,1)
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
        sol = solve_ivp(lambda t, s: problem.ode_func(t, s, data, thr),
                        (t_start, t_end), full0, t_eval=np.linspace(t_start, t_end, n_points),
                        dense_output=False,
                        method="LSODA")
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
        self.setWindowTitle("Planar OCP - Free Final Time (Shooting, H(tf)=0)")
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
        # (A) First row: costate guess boxes, time, "Integrate & Plot", "Shoot Single", "Shoot Multi"
        # -------------------------------------------------------------
        top_layout = QHBoxLayout()
        self.main_layout.addLayout(top_layout)
        
        ttop = QHBoxLayout()
        ttop.addWidget(QLabel("Select Problem:"))

        self.combo_problem = QComboBox()
        # For each problem in self.problems, add an item:
        for p in self.problems:
            # if each Problem has a .name() method, you can do:
            self.combo_problem.addItem(p.name())
            # or, if not, just do self.combo_problem.addItem("Planar"), etc.
        # Connect the signal to your callback:
        self.combo_problem.currentIndexChanged.connect(self.on_problem_changed)
        ttop.addWidget(self.combo_problem)

        # Spin exponent for costate steps
        vbox_step = QVBoxLayout()
        top_layout.addLayout(vbox_step)
        vbox_step.addWidget(QLabel("Step exponent (costates)"))
        self.spin_exp = QSpinBox()
        self.spin_exp.setRange(-6, 6)
        self.spin_exp.setValue(0)
        self.spin_exp.valueChanged.connect(self.on_step_changed)
        vbox_step.addWidget(self.spin_exp)

        # add combo problem
        top_layout.addLayout(ttop)

        # costates
        vbox_costates = QVBoxLayout()
        top_layout.addLayout(vbox_costates)
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        vbox_costates.addLayout(row1)
        vbox_costates.addLayout(row2)

        self.spin_lx = QDoubleSpinBox(); row1.addWidget(QLabel("lr0:")); row1.addWidget(self.spin_lx)
        self.spin_ly = QDoubleSpinBox(); row1.addWidget(QLabel("lth0:")); row1.addWidget(self.spin_ly)
        self.spin_lz = QDoubleSpinBox(); row1.addWidget(QLabel("lph0:")); row1.addWidget(self.spin_lz)
        self.spin_lvx= QDoubleSpinBox();row2.addWidget(QLabel("lu0:"));row2.addWidget(self.spin_lvx)
        self.spin_lvy= QDoubleSpinBox();row2.addWidget(QLabel("lv0:"));row2.addWidget(self.spin_lvy)
        self.spin_lvz= QDoubleSpinBox();row2.addWidget(QLabel("lw0:"));row2.addWidget(self.spin_lvz)
        self.spin_lm = QDoubleSpinBox(); row2.addWidget(QLabel("lm0:")); row2.addWidget(self.spin_lm)

        # switch times
        self.spin_t1 = QDoubleSpinBox()
        self.spin_t1.setRange(0.1, 1e6)
        self.spin_t1.setValue(0.0)
        row1.addWidget(QLabel("t1:")); row1.addWidget(self.spin_t1)

        self.spin_t2 = QDoubleSpinBox()
        self.spin_t2.setRange(0.1, 1e6)
        self.spin_t2.setValue(0.0)
        row2.addWidget(QLabel("t2:")); row2.addWidget(self.spin_t2)

        # final time guess
        self.spin_tf = QDoubleSpinBox()
        self.spin_tf.setRange(0.1, 1e6)
        self.spin_tf.setValue(20.0)
        row2.addWidget(QLabel("Time guess:")); row2.addWidget(self.spin_tf)

        # Set ranges/precision
        self.costate_spins = [self.spin_lx, self.spin_ly, self.spin_lz, self.spin_lvx, self.spin_lvy, self.spin_lvz, self.spin_lm]
        for sp in self.costate_spins:
            sp.setRange(-1e6, 1e6)
            sp.setDecimals(8)
            sp.setValue(0.0)
            sp.valueChanged.connect(self.on_plot)

        self.spin_t1.valueChanged.connect(self.on_plot)
        self.spin_t2.valueChanged.connect(self.on_plot)
        self.spin_tf.valueChanged.connect(self.on_plot)

        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self.on_plot)
        top_layout.addWidget(self.btn_plot)

        self.btn_shoot_single = QPushButton("Shoot - Single Step")
        self.btn_shoot_single.clicked.connect(self.on_shoot_single)
        top_layout.addWidget(self.btn_shoot_single)

        self.btn_shoot_multi = QPushButton("Shoot - Multiple Steps")
        self.btn_shoot_multi.clicked.connect(self.on_shoot_multi)
        top_layout.addWidget(self.btn_shoot_multi)

        self.spin_multistep = QSpinBox()
        self.spin_multistep.setRange(1, 5000)
        self.spin_multistep.setValue(10)   # default 10
        top_layout.addWidget(QLabel("Multi-steps:"))
        top_layout.addWidget(self.spin_multistep)

        # -------------------------------------------------------------
        # (B) Second row: alpha, delta, reset
        # -------------------------------------------------------------
        second_layout = QHBoxLayout()
        self.main_layout.addLayout(second_layout)

        # alpha (relax factor)
        second_layout.addWidget(QLabel("alpha (relax factor):"))
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(1e-9, 1e2)
        self.spin_alpha.setDecimals(8)
        self.spin_alpha.setValue(1e-6)
        second_layout.addWidget(self.spin_alpha)

        # delta (FD step)
        second_layout.addWidget(QLabel("delta (FD step) caution below 1e-6:"))
        self.spin_delta = QDoubleSpinBox()
        self.spin_delta.setRange(1e-9, 1e-2)
        self.spin_delta.setDecimals(7)
        self.spin_delta.setValue(1e-7)
        second_layout.addWidget(self.spin_delta)

        # reset button
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        second_layout.addWidget(self.btn_reset)

        # A button to save the data to JSON
        self.btn_save = QPushButton("Save JSON")
        self.btn_save.clicked.connect(self.on_save_json)
        second_layout.addWidget(self.btn_save)

        second_layout.addWidget(QLabel("R0 (dist scale):"))
        self.spin_R0 = QDoubleSpinBox()
        self.spin_R0.setRange(1e-9, 1e9)
        self.spin_R0.setDecimals(6)
        self.spin_R0.setValue(6371.0)  # example default
        second_layout.addWidget(self.spin_R0)

        second_layout.addWidget(QLabel("V0 (vel scale):"))
        self.spin_V0 = QDoubleSpinBox()
        self.spin_V0.setRange(1e-9, 1e9)
        self.spin_V0.setDecimals(6)
        self.spin_V0.setValue(7.909788)   # example default
        second_layout.addWidget(self.spin_V0)

        self.btn_enable_nd = QPushButton("Enable ND")
        self.btn_enable_nd.clicked.connect(self.on_enable_nd)
        second_layout.addWidget(self.btn_enable_nd)

        self.btn_disable_nd = QPushButton("Disable ND")
        self.btn_disable_nd.clicked.connect(self.on_disable_nd)
        second_layout.addWidget(self.btn_disable_nd)

        self.label_fdmethod = QLabel("FD Method:")
        second_layout.addWidget(self.label_fdmethod)

        self.combo_fdmethod = QComboBox()
        self.combo_fdmethod.addItem("forward")
        self.combo_fdmethod.addItem("backward")
        self.combo_fdmethod.addItem("central")
        second_layout.addWidget(self.combo_fdmethod)

        self.chk_timefree = QCheckBox("Time is Free?")
        self.chk_timefree.setChecked(self.timefree)  # initial state
        self.chk_timefree.toggled.connect(self.on_timefree_toggled)
        second_layout.addWidget(self.chk_timefree)

        self.use_nondim = False

        # -------------------------------------------------------------
        # (C) Third row: updated param (read-only for costates + final time?)
        # -------------------------------------------------------------
        third_layout = QHBoxLayout()
        self.main_layout.addLayout(third_layout)

        lbl_upd = QLabel("UPDATED param (costates + tf):")
        third_layout.addWidget(lbl_upd)

        self.upd_lx = QDoubleSpinBox(); third_layout.addWidget(QLabel("lr_upd:")); third_layout.addWidget(self.upd_lx)
        self.upd_ly = QDoubleSpinBox(); third_layout.addWidget(QLabel("lth_upd:")); third_layout.addWidget(self.upd_ly)
        self.upd_lz = QDoubleSpinBox(); third_layout.addWidget(QLabel("lph_upd:")); third_layout.addWidget(self.upd_lz)        
        self.upd_lvx= QDoubleSpinBox(); third_layout.addWidget(QLabel("lu_upd:"));third_layout.addWidget(self.upd_lvx)
        self.upd_lvy= QDoubleSpinBox(); third_layout.addWidget(QLabel("lv_upd:"));third_layout.addWidget(self.upd_lvy)
        self.upd_lvz= QDoubleSpinBox(); third_layout.addWidget(QLabel("lw_upd:"));third_layout.addWidget(self.upd_lvz)        
        self.upd_lm = QDoubleSpinBox(); third_layout.addWidget(QLabel("lm_upd:")); third_layout.addWidget(self.upd_lm)
        self.upd_t1 = QDoubleSpinBox(); third_layout.addWidget(QLabel("t1_upd:")); third_layout.addWidget(self.upd_t1)
        self.upd_t2 = QDoubleSpinBox(); third_layout.addWidget(QLabel("t2_upd:")); third_layout.addWidget(self.upd_t2)
        self.upd_tf = QDoubleSpinBox(); third_layout.addWidget(QLabel("tf_upd:")); third_layout.addWidget(self.upd_tf)

        self.updated_spins = [self.upd_lx, self.upd_ly, self.upd_lz, self.upd_lvx, self.upd_lvy, self.upd_lvz, self.upd_lm, self.upd_t1, self.upd_t2, self.upd_tf]
        for sp in self.updated_spins:
            sp.setRange(-1e8, 1e8)
            sp.setDecimals(6)
            sp.setValue(0.0)
            sp.setReadOnly(True)  # user cannot manually edit updated values

        # -------------------------------------------------------------
        # (D) The embedded Matplotlib figure
        # -------------------------------------------------------------
        self.canvas = MplCanvas(self, width=12, height=6, dpi=100)
        self.main_layout.addWidget(self.canvas)
        self.axes = self.canvas.fig.subplots(4,4).flatten()               # modifico subplots(3,4) in subplots(4,4) per far comparire 
        for ax in self.axes:                                              # un'ulteriore riga di grafici (i plot da 12 diventano 16) 
            ax.grid(True)

        # Rocket parameters
        self.T_max = 0.025/(TN*10**3)
        self.u_e   = 20000.0/(V*10**3)               #ho aggiunto 10^3 ai denominatori per far quadrare le dimensioni
        self.lam_initial = None  # store initial param guess for "Reset"
        self.newton_iter = 0
        self.update_spin_steps()

    ####################################################################
    #   Utility: read/write param guess
    ####################################################################
    def on_timefree_toggled(self, checked):
        """
        Called whenever the user checks or unchecks the "Time is Free?" box.
        `checked` is True if the box is checked, otherwise False.
        """
        self.timefree = checked
    
    def on_enable_nd(self):
        """
        Turn on nondimensional integration, based on spin_R0 and spin_V0.
        """
        self.use_nondim = True
        print("Nondimensional mode ENABLED.")

    def on_disable_nd(self):
        """
        Turn off nondimensional integration (go back to dimensional).
        """
        self.use_nondim = False
        print("Nondimensional mode DISABLED.")
    
    def on_save_json(self):
        """
        Save a JSON file 'output.json' with:
        - T_max, u_e
        - alpha, delta
        - lam_initial (the first time user pressed 'Integrate & Plot')
        - lam_updated (the last known 'updated' param)
        """
        import json

        # Gather the data
        data_dict = {}

        # Basic rocket params
        data_dict["T_max"] = self.T_max
        data_dict["u_e"]   = self.u_e
        

        # Current alpha, delta
        data_dict["alpha"] = self.spin_alpha.value()
        data_dict["delta"] = self.spin_delta.value()

        # The initial guess, if we have one
        if self.lam_initial is not None:
            data_dict["lam_initial"] = {
                "lx0":  self.lam_initial[0],
                "ly0":  self.lam_initial[1],
                "lz0":  self.lam.initial[2],
                "lvx0": self.lam_initial[3],
                "lvy0": self.lam_initial[4],
                "lvz0": self.lam_initial[5],
                "lm0":  self.lam_initial[6],
                "tf0":  self.lam_initial[7] if len(self.lam_initial) == 8 else None
            }
        else:
            data_dict["lam_initial"] = "No initial guess stored yet"

        # The currently displayed 'updated' param in the read-only spins
        # (including final time if you added that as well).
        lam_upd = [
            self.upd_lx.value(),
            self.upd_ly.value(),
            self.upd_lz.value(),
            self.upd_lvx.value(),
            self.upd_lvy.value(),
            self.upd_lvz.value(),
            self.upd_lm.value()
        ]
        # If you have an updated final time spin: self.upd_tf
        lam_upd.append(self.upd_tf.value())

        data_dict["lam_updated"] = {
            "lx":  lam_upd[0],
            "ly":  lam_upd[1],
            "lz":  lam_upd[2],         
            "lvx": lam_upd[3],
            "lvy": lam_upd[4],
            "lvz": lam_upd[5],
            "lm":  lam_upd[6],
            "tf":  lam_upd[7]
        }

        # Just an example: store the iteration count
        data_dict["newton_iter"] = self.newton_iter

        # Write to JSON
        with open("output.json", "w") as f:
            json.dump(data_dict, f, indent=2)

        print("Saved current data to 'output.json'.")
    
    def read_param_guess(self):
        """
        Return 8D param = [lx0, ly0, lz0, lvx0, lvy0, lvz0, lm0, tf].
        We'll use the spin_time as the last entry.
        """
        return [
            self.spin_lx.value(),
            self.spin_ly.value(),
            self.spin_lz.value(),
            self.spin_lvx.value(),
            self.spin_lvy.value(),
            self.spin_lvz.value(),
            self.spin_lm.value(),
            self.spin_t1.value(),
            self.spin_t2.value(),
            self.spin_tf.value()
        ]

    def write_updated_param(self, param):
        """
        Write param = [lx, ly, lvx, lvy, lm, tf]
        into the read-only spin boxes. We also update the main spin_tf so user sees new guess.
        """
        self.upd_lx.setValue(param[0])
        self.upd_ly.setValue(param[1])
        self.upd_lz.setValue(param[2])
        self.upd_lvx.setValue(param[3])
        self.upd_lvy.setValue(param[4])
        self.upd_lvz.setValue(param[5])
        self.upd_lm.setValue(param[6])
        self.upd_t1.setValue(param[7])
        self.upd_t2.setValue(param[8])
        if self.timefree:
            self.upd_tf.setValue(param[9])
            self.spin_tf.setValue(param[9])

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
    def plot_colored_segments(self, ax, t, y, sf, default_color="C0", highlight_color="orange"):
        indices = np.where(np.diff(np.sign(sf)))[0]  # Find sign-change indices
        segments = np.split(np.arange(len(t)), indices + 1)  # Split indices into segments

        for seg in segments:
            color = highlight_color if sf[seg[0]] > 0 else default_color
            ax.plot(t[seg], y[seg], color=color)

    def plot_trajectory(self, guesses, data, tfs):
        if not self.trueplot:
            return

        t_array, states, sf_array, H_array = integrate_with_sf(
            self.current_problem, guesses, data, tfs)

        # Make a single array
        t_array = np.concatenate(t_array)
        states = np.concatenate(states, axis=1)

        for ax in self.axes:
            ax.clear()
            ax.grid(True)

        labels = (
            self.current_problem.state_labels() +
            self.current_problem.costate_labels() +
            ["SF(t)", "H(t)"]
        )

        # Loop through states and costates with proper segment coloring
        for i in range(len(labels) - 2):
            self.plot_colored_segments(self.axes[i], t_array, states[i, :], sf_array)
            self.axes[i].set_title(rf"${labels[i]}$")

        # SF(t) plot
        self.plot_colored_segments(self.axes[14], t_array, sf_array, sf_array)      # Cabio l'indice di self.axes (da [10]/[11] a [14]/[15]) sia per SF 
        self.axes[14].set_title("SF(t)")                                            # che H in modo da avere la label giusta sui plot 3D

        # Hamiltonian plot
        self.plot_colored_segments(self.axes[15], t_array, H_array, sf_array)
        self.axes[15].set_title("Hamiltonian(t)")

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    ####################################################################
    #   Newton iteration
    ####################################################################
    def do_one_newton_iteration(self, guesses):
        alpha = self.spin_alpha.value()
        delta = self.spin_delta.value()
        data = [self.T_max, self.u_e, delta]

        fd_method = self.combo_fdmethod.currentText()
        # One iteration
        param_next, err, norm_err = newton_shoot_step(
            self.current_problem,
            guesses, data, alpha=alpha, preverr=self.preverr,
            method=fd_method, timefree=self.timefree)
        self.preverr = norm_err
        self.newton_iter += 1
        print(f"Iter {self.newton_iter:4d}")
        print("Guess: ", end="")
        for i, p in enumerate(guesses):
            print(f"{p: .4e} ", end="")
        print("\nNew g: ", end="")
        for i, p in enumerate(param_next):
            print(f"{p: .4e} ", end="")
        print(f"({np.linalg.norm(err): .4e}, {max(abs(err)):.4e})\nErr:   ", end="")   # aggiunto il valore assoluto (abs(err)) in modo che stampi 
        for i, e in enumerate(err):                                                    # il modulo del max errore residuo e non il massimo positivo
            print(f"{e: .4e} ", end="")
        print()

        # Update the read-only boxes
        self.write_updated_param(param_next)

        # Re-plot using the updated param
        NA = 1
        if self.current_problem.switching_structure is not None:
            NA = len(self.current_problem.switching_structure)
        lambdas = param_next[:-NA]
        tfs   = param_next[-NA:]
        self.plot_trajectory(lambdas, data, tfs)

        return param_next, err, norm_err

    ####################################################################
    #   Button Callbacks
    ####################################################################
    def on_problem_changed(self, index):
        self.current_problem = self.problems[index]
    
    def on_plot(self):
        sender = self.sender()
        if sender is None:
            return

        if isinstance(sender, QDoubleSpinBox):
            if sender.hasFocus():  # Skip plotting if the spinbox has focus (manual typing)
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
        return [
            self.upd_lx.value(),
            self.upd_ly.value(),
            self.upd_lz.value(),
            self.upd_lvx.value(),
            self.upd_lvy.value(),
            self.upd_lvz.value(),
            self.upd_lm.value(),
            self.upd_t1.value(),
            self.upd_t2.value(),
            self.upd_tf.value()
        ]

    def on_shoot_single(self):
        param_current = self.read_param_guess()
        if any(sp.value() != 0.0 for sp in self.updated_spins):
            param_current = self.read_updated_param()
        self.trueplot = True
        self.do_one_newton_iteration(param_current) # always plot single step

    def on_shoot_multi(self):
        param_current = self.read_param_guess()
        if any(sp.value() != 0.0 for sp in self.updated_spins):
            param_current = self.read_updated_param()
        max_iter = self.spin_multistep.value()  # read from the new spin box
        for k in range(max_iter):
            self.trueplot = k % self.plot_every == 0 or k == max_iter-1
            param_next, err, norm_err = self.do_one_newton_iteration(param_current)
            param_current = param_next
            QApplication.processEvents()
            if  convergence_fun(err, norm_err):
                print("Converged!")
                break

    def on_reset(self):
        """
        Reset the top param to the original guess we stored
        the first time user pressed "Integrate & Plot".
        Also reset updated param to zeros, iteration=0.
        """
        self.trueplot = True
        if self.lam_initial is not None:
            # restore top guess
            lx0, ly0, lz0, lvx0, lvy0, lvz0, lm0, *ts = self.lam_initial
            self.spin_lx.setValue(lx0)
            self.spin_ly.setValue(ly0)
            self.spin_lz.setValue(lz0)
            self.spin_lvx.setValue(lvx0)
            self.spin_lvy.setValue(lvy0)
            self.spin_lvz.setValue(lvz0)
            self.spin_lm.setValue(lm0)
            NA = 1
            if self.current_problem.switching_structure is not None:
                NA = len(self.current_problem.switching_structure)
            self.spin_t1.setValue(ts[0])
            self.spin_t2.setValue(ts[1])
            self.spin_tf.setValue(ts[-1])

        # clear updated param
        for sp in self.updated_spins:
            sp.setValue(0.0)
        self.newton_iter = 0
        print("Reset done: top param back to initial, updated param = 0")


def main():
    app = QApplication(sys.argv)
    gui = PlanarGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
