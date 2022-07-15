import numpy as np

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
