from numba import njit

@njit
def adimensionalizing_mu(MU, R = 6371, V = 7.909788019132536):
    
    return MU/(R*V**2)
    
    
    
    