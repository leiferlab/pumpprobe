import numpy as np
import pumpprobe as pp

def dyson(h,time=None,omega=None,domain_in="omega",domain_out=None,
           direction="direct",return_all=False,*args,**kwargs):
    '''Solves the Dyson equation
    $$ F_{ij} = f_{ij} + \\sum\\limits_{\\mu\\neq j} f_{i\\mu}*F_{\\mu j}$$
    to either return F given f (direction=\"direct\"), or to return f given F
    (direction=\"inverse\").
    
    Parameters
    ----------
    h: numpy ndarray
        Input matrix (f or F, depending on direction). Indexed as
        h[i,j,omega] or h[i,j,t], depending on domain_in.
    time: numpy ndarray (optional)
        Time axis. Required if domain_in or domain_out are \"time\". 
        Default: None.
    omega: numpy ndarray (optional)
        Frequency axis. Required if domain_in is \"omega\" and domain_out is 
        \"time\". Default: None.
    domain_in: str (optional)
        Input domain. Possible values: omega, time. Default: omega.
    domain_out: str (optional)
        Output domain. Possible values: omega, time, None. If None, domain_out
        is set the same as domain_in. Default: None.
    direction: str (optional)
        Determines whether the function returns F given f (direct) or f given
        F (inverse). Default: direct.
    return_all: bool (optional)
        Whether to return the omega or time axis.
        
    Returns
    -------
    g: numpy ndarray
        Output matrix (f or F, depending on direction). Indexed as
        h[i,j,omega] or h[i,j,t], depending on domain_out.
    '''       
    
    n = h.shape[0]
    m = h.shape[1]
    
    # Determine desired output domain
    if domain_out is None: domain_out = domain_in
    
    if domain_out=="time" and time is None:
        raise ValueError("If domain_out is set to \"time\", you need to "+\
                         "provide the time axis too.")
                         
    if domain_out=="time" and domain_in=="omega" and omega is None:
        raise ValueError("If domain_out is set to \"time\", and domain_in is"+\
                         " \"omega\" you need to provide both the time axis"+\
                         "and the omega axis.")
    
    if domain_in=="time":
        # F passed as a function of time
        if time is None:
            raise ValueError("If F is passed in the time domain,"+\
                             "you need to provide the time axis too.")
        else:
            # Compute the Fourier transform of F
            if "log_upsample" in kwargs.keys():
                l_u = kwargs["log_upsample"]
            else:
                l_u = 2
            _, omega = pp.ft_cubic(h[0,0],time,log_upsample=l_u)
            n_omega = omega.shape[0]
            h_w = np.zeros((n,m,n_omega),dtype=np.complex128)

            for i in np.arange(n):
                for j in np.arange(m):
                    h_w[i,j,:], _ = pp.ft_cubic(h[i,j],time,log_upsample=l_u)
    else:
        # h passed as a function of omega
        h_w = h
        n_omega = h_w.shape[-1]
    
    # Solve the linear system
    h_w = np.moveaxis(h_w,-1,0)
    
    # The coefficient matrix is always the same for the inverse problem, but it
    # is different for every j in the direct problem.
    if direction=="inverse":
        # For the inverse problem, make a stack of matrices for each omega 
        # (along the 0th axis).
        a = np.copy(h_w)
        for i in np.arange(min(n,m)): a[:,i,i] = 1.0
        b = h_w
        g_w = np.linalg.solve(a,b)

    elif direction=="direct":
        # For the direct problem, the system has different coefficients for
        # each j (second index in Fij). 
        g_w = np.zeros((n_omega,n,m),dtype=np.complex128)
        a = np.zeros((n_omega,n,m),dtype=np.complex128)
        b = np.zeros((n_omega,n),dtype=np.complex128)
        for j in np.arange(m):
            a = -h_w.copy()
            b = h_w[:,:,j].copy()
            a[:,:,j] = 0.0
            # Restore the jth entry for the F_jj equation.
            a[:,j,j] = -h_w[:,j,j]
            for i in np.arange(m): a[:,i,i] += 1.0
            #g_w[:,:,j] = np.linalg.solve(a,b)
            print("USING LSTSQ IN DYSON")
            g_w[0,:,j],_,_,_ = np.linalg.lstsq(a[0],b[0])
                
    g_w = np.moveaxis(g_w,0,-1)
        
    if domain_out == "time":
        if "log_upsample_ift" in kwargs.keys():
                l_u_i = kwargs["log_upsample_ift"]
        else:
            l_u_i = 0
        # Get the time axis with the set upsampling.
        _,t = pp.ift_cubic_real(g_w[0,0],omega,log_upsample=l_u_i)
        g = np.zeros((n,m,time.shape[0]))
        for i in np.arange(n):
            for j in np.arange(m):
                # Get the inverse Fourier transform.
                tmp_g,_ = pp.ift_cubic_real(g_w[i,j],omega)
                # Interpolate on the input time axis.
                g[i,j] = np.interp(time,t,tmp_g)
        if return_all:
            return g, t
        else:
            return g
    else:
        if return_all:
            return g_w, omega
        else:
            return g_w
