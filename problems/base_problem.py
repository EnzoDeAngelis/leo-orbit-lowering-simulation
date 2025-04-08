# file: problems/base_problem.py
from abc import ABC, abstractmethod
import numpy as np

class BaseProblem(ABC):
    """
    Abstract base class for describing an OCP problem that the GUI can handle.
    Each problem must define:
      - dimension of (states + costates)
      - ODE function
      - boundary error function
      - (optionally) H function, etc.
    """

    @abstractmethod
    def name(self):
        """Return a short string name, e.g. 'Planar' or 'Orbital'."""
        pass

    @abstractmethod
    def default_param_guess(self):
        """
        Return a default guess for costates (and final time if free).
        For example, an array of length 6: [lx0, ly0, lvx0, lvy0, lm0, tf0].
        """
        pass

    @abstractmethod
    def ode_func(self, t, state, data):
        """
        The ODE (states+costates) = f(t, state, data).
        Typically shape (2*n + 1) or (2*n) depending on how you pack mass + costates, etc.
        """
        pass

    @abstractmethod
    def boundary_error(self, param_guess, data, timefree):
        """
        Evaluate boundary conditions for shooting method, e.g. 
        final x, y, vx, vy, Hamiltonian, etc. Must return 1D array of mismatches.
        """
        pass

    @abstractmethod
    def compute_H(self, state, data):
        """Compute the Hamiltonian at a given state if you need it for H(tf)=0 conditions."""
        pass

    @abstractmethod
    def compute_sf(self, state, data):
        """Compute the switching function at a given state if you need it for switching conditions."""
        pass

    @abstractmethod
    def states(self):
        """How many states (not counting costates) does this problem have?"""
        pass

    @abstractmethod
    def costates(self):
        """How many costates does this problem have?"""
        pass

    @abstractmethod
    def state_labels(self):
        """Return a list of strings for labeling each of the states (for plotting)."""
        pass

    @abstractmethod
    def costate_labels(self):
        """Return a list of strings for labeling each costate (for plotting)."""
        pass
