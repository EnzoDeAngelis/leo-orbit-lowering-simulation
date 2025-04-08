import pytest
from problems.orb2d_problem import swfun
import numpy as np

# NOTA - Il file può essere rinominato, ma deve iniziare come test_*.py

def test_simple():
    assert 1 == 1
    
def test_swfun():
    fullstate = np.ones(10)
    data = [1, 2000, 5]
    SF, LV = swfun(fullstate, data)
    assert SF
    assert LV
    assert LV >= 0, 'Il primer vector deve essere positivo'
    fullstate[4] = 0
    with pytest.raises(ZeroDivisionError):
        swfun(fullstate, data)
    