import numpy as np
import pumpprobe as pp

def integral_c(A, delta=1., k=8):
    if A.dtype==np.complex128:
        I = pp.integral(A.real.copy(),delta,k) 
        I += 1.0j*pp.integral(A.imag.copy(),delta,k)
    elif A.flags['C_CONTIGUOUS']==False: 
        I = pp.integral(A.copy(),delta,k)
    else:
        I = pp.integral(A,delta,k)
        
    return I    
