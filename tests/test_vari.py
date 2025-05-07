import pytest
from problems.orb3d_problem import swfun, ode_orb3d, compute_H_orb3d
import numpy as np

# NOTA - Il file può essere rinominato, ma deve iniziare come test_*.py

def test_swfun():
    fullstate = np.ones(14)
    data = [1, 1000, 5]
    SF, LV = swfun(fullstate, data)
    assert np.isfinite(SF), "SF non è un numero valido"
    assert np.isfinite(LV), "LV non è un numero valido"
    assert LV >= 0, 'Il primer vector deve essere positivo'
    
    data[1] = 0
    with pytest.raises(ZeroDivisionError):
       swfun(fullstate, data)
    
def test_ode_orb3d():
    fullstate = np.ones(14)
    data = [1, 1000, 5]
    t = 0.0
    dfdt = ode_orb3d(t, fullstate, data, None)
    
    assert len(dfdt) == len(fullstate), "Lunghezza dfdt diversa da fullstate" 

def test_compute_H_orb3d():
    fullstate = np.ones(14)
    data = [1, 1000, 5]
    H = compute_H_orb3d(fullstate, data, None)
    assert np.isfinite(H), "H non è un numero valido"

    
    