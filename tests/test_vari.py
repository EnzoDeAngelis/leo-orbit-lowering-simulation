import pytest
from problems.orb2d_problem import swfun, ode_orb2d, compute_H_orb2d
import numpy as np

# NOTA - Il file può essere rinominato, ma deve iniziare come test_*.py

""" def test_simple():
    assert 1 == 1 
"""
    
def test_swfun():
    fullstate = np.ones(10)
    data = [1, 1000, 5]
    SF, LV = swfun(fullstate, data)
    assert np.isfinite(SF), "SF non è un numero valido"
    assert np.isfinite(LV), "LV non è un numero valido"
    #assert LV >= 0, 'Il primer vector deve essere positivo'
    #fullstate[4] = 0
    #with pytest.raises(ZeroDivisionError):
    #    swfun(fullstate, data)
    
"""def test_ode_orb2d():
    fullstate = np.ones(10)
    data = [1, 1000, 5]
    dfdt = ode_orb2d(1, fullstate, data, None)
    
    assert len(dfdt) == len(fullstate), "Lunghezza dfdt diversa da fullstate" 

def test_compute_H_orb2d():
    fullstate = np.ones(10)
    data = [1, 1000, 5]
    H = compute_H_orb2d(fullstate, data, None)
    assert np.isfinite(H), "H non è un numero valido"
"""
    
    