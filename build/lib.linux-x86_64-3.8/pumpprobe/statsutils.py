import numpy as np
from scipy.optimize import curve_fit

'''Weighted correlation coefficient, Wikipedia and [1].
[1] https://stackoverflow.com/questions/38641691/weighted-correlation-coefficient-with-pandas'''

def m(x, w):
    """Weighted Mean"""
    return np.sum(x * w) / np.sum(w)

def cov(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

def weighted_corr(x, y, w):
    """Weighted Correlation"""
    return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))
    
def pearsonr_sample(x,y):
    '''Pearson correlation coefficient for a sample'''
    avgx = np.average(x)
    avgy = np.average(y)
    
    xmavgx = x-avgx
    ymavgy = y-avgy
    
    a = np.sum( (xmavgx) * (ymavgy) )
    b = np.sum( np.power(xmavgx,2) )
    c = np.sum( np.power(ymavgy,2) )
    
    r = a/np.sqrt(b)/np.sqrt(c)
    
    return r
    
def R2(a,b):
    A, B = np.array([a,]).T,b
    par, res, _, _ = np.linalg.lstsq(A,B,rcond=None)
    m, = par; c = 0
    avg_B = np.average(B)
    SSres = np.sum( np.power(B - (m*A.T[0]+c),2) )
    SStot = np.sum( np.power(B - avg_B,2) )
    R2 = 1.0 - SSres/SStot
    
    return R2
    
def line(a,m):
    return a*m

def R2nl(a,b):
    popt,_ = curve_fit(line,a,b,p0=[1.])
    m, = popt    
    c = 0
    avg_b = np.average(b)
    SSres = np.sum( np.power(b - (m*a+c),2) )
    SStot = np.sum( np.power(b - avg_b,2) )
    R2 = 1.0 - SSres/SStot
    
    return R2
    
def p_to_stars(p):
    if p<=0.0001: stars = "****"
    elif p<=0.001: stars="***"
    elif p<=0.01: stars="**"
    elif p<=0.05: stars="*"
    else: stars = "n.s."
    
    return stars
