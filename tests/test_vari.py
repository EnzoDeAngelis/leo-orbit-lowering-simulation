import pytest
from OCP.functions import swfun, ode_orb3d, compute_H_orb3d
from OCP.perturbations import compute_drag_acceleration_zen, load_density_data
from OCP.pert_derivatives import compute_drag_partials_extended
import numpy as np

# NOTA - Il file può essere rinominato, ma deve iniziare come test_*.py
altitude_array, density_array = load_density_data()

def test_compute_drag_acceleration_zen(): 
    th, ph, u, v, w, m = np.ones(6)
    r = 1.00147227463
    a_drag = compute_drag_acceleration_zen(r, th, ph, u, v, w, m, altitude_array, density_array)
    a_drag_x = a_drag[0]
    
    expected = round(-2*0.008316572278407607, 5)  # -> -0.00832 con A = 1, se A = 0.5 tolgo il x2
    assert round(a_drag_x, 5) == expected, f"Valore ottenuto: {a_drag_x}, atteso: -0.00832" 


def test_swfun():
    fullstate = np.ones(14)
    data = np.array([1, 1000, 5])
    SF, LV = swfun(fullstate, data)
    assert np.isfinite(SF), "SF non è un numero valido"
    assert np.isfinite(LV), "LV non è un numero valido"
    assert LV >= 0, 'Il primer vector deve essere positivo'
    
    data[1] = 0
    with pytest.raises(ZeroDivisionError):
       swfun(fullstate, data)


def test_ode_orb3d():
    fullstate = np.ones(14)
    data = np.array([1, 1000, 5])
    t = 0.0
    dfdt = ode_orb3d(t, fullstate, data, None)
    
    assert fullstate[6] > 0, 'La massa deve essere maggiore di 0'
    assert len(dfdt) == len(fullstate), "Lunghezza dfdt diversa da fullstate"
    
    fullstate_mass_zero = fullstate.copy()
    fullstate_mass_zero[6] = 0
    with pytest.raises(ZeroDivisionError):
        ode_orb3d(t, fullstate_mass_zero, data)
        
    data_ue_zero = data.copy()
    data_ue_zero[1] = 0
    with pytest.raises(ZeroDivisionError):
        ode_orb3d(t, fullstate, data_ue_zero) 


def test_compute_H_orb3d():
    fullstate = np.ones(14)
    data = np.array([1, 1000, 5])
    H = compute_H_orb3d(fullstate, data, None)
    assert np.isfinite(H), "H non è un numero valido"


def test_compute_drag_partials():
    r, th, ph, u, v, w, m = np.ones(7)
    J = compute_drag_partials_extended(r, th, ph, u, v, w, m, altitude_array, density_array)
    
    assert J is not None
