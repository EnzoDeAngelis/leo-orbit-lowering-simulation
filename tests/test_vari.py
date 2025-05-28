import pytest
from OCP.gui import convergence_fun
import numpy as np

# NOTA - Il file può essere rinominato, ma deve iniziare come test_*.py

def test_convergence_fun():
    # Caso 1: Tutti gli errori molto piccoli → converge per max(abs(err))
    # Caso 2: Un errore sopra la soglia max, ma norma sotto soglia → converge per norma
    # In questo esempio ---> max = 1e-5 > 1e-6, ma norma < 1e-4
    err = np.array([1e-6, -5e-9, 2e-10, -1e-7, 9e-8, 3e-9, -4e-10, 1e-8, 2e-9, -7e-8])
    norm_err = np.linalg.norm(err)
    assert convergence_fun(err, norm_err) == True, "Uno o più errori sono maggiori della soglia di convergenza"

    
    # Caso 3: Tutti gli errori grandi → non converge
    err = np.array([1e-3, -2e-3, 5e-4, -3e-4, 4e-4, -6e-4, 7e-4, -1e-3, 2e-3, -3e-3])
    norm_err = np.linalg.norm(err)
    assert convergence_fun(err, norm_err) == False, "Tutti gli errori residui sono minori della soglia, si ha convergenza"
    


    
    

    
    