import numpy as np, matplotlib.pyplot as plt
from scipy.signal import savgol_coeffs
import pumpprobe as pp, wormdatamodel as wormdm

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
folders = np.loadtxt(ds_list, dtype='str', delimiter="\n")
n_ds = len(folders)

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,                 
                 "matchless_nan_th_from_file": True}

nan_th = 0.3
dff = np.zeros(0)
dff2 = np.zeros(0)
sd = np.zeros(0)

for i_ds in np.arange(n_ds):
    folder = "/".join(folders[i_ds].split("/")[:-1])+"/"
    fconn = pp.Fconn.from_file(folder)
    sig = wormdm.signal.Signal.from_file(folder,"green",**signal_kwargs)
    
    sderker = savgol_coeffs(13, 2, deriv=2, delta=fconn.Dt)
    
    sder = np.zeros_like(sig.data)            
    for k in np.arange(sig.data.shape[1]):
        sder[:,k] = np.convolve(sderker,sig.data[:,k],mode="same")
    
    dff_ = np.zeros((fconn.n_stim,fconn.n_neurons))
    dff2_ = np.zeros((fconn.n_stim,fconn.n_neurons))
    sd_ = np.zeros((fconn.n_stim,fconn.n_neurons))
    
    for ie in np.arange(fconn.n_stim):
        i0 = fconn.i0s[ie]
        i1 = fconn.i1s[ie]
        shift_vol = fconn.shift_vol
        y = sig.get_segment(i0,i1,baseline=False,normalize="none")
        nan_mask = sig.get_segment_nan_mask(i0,i1)
        
        for ineu in np.arange(fconn.n_neurons):
            if np.sum(nan_mask[:,ineu])>nan_th*len(y[:,ineu]):
                dff_[ie,ineu] = 0.0
                dff2_[ie,ineu] = 0.0
                sd_[ie,ineu] = 0.0
            else:
                
                pre = np.average(y[:shift_vol,ineu])                                 
                dy = np.average( y[shift_vol:,ineu] - pre )/pre
                
                sd__ = np.sum(sder[i0+shift_vol-5:i0+shift_vol+6,ineu])
                    
                if ineu not in fconn.resp_neurons_by_stim[ie]:
                    dff_[ie,ineu] = 0.0
                else:
                    dff_[ie,ineu] = dy
                    
                dff2_[ie,ineu] = dy
                sd_[ie,ineu] = sd__
                
    dff = np.append(dff,np.ravel(dff_))
    dff2 = np.append(dff2,np.ravel(dff2_))
    sd = np.append(sd,np.ravel(sd_))
    
ctrl_dff,ctrl_dff2,ctrl_sd = pp.Funatlas.get_ctrl_distributions()

fig = plt.figure(1)
ax = fig.add_subplot(111)
#ax.hist(dff2,50,range=(-0.4,0.4))
ax.hist(dff,50,range=(-0.4,0.4))
ax.set_xlabel("dFF")
ax.set_ylabel("n")

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.hist(sd,50)
ax.set_xlabel("second derivative")
ax.set_ylabel("n")

fig = plt.figure(3)
ax = fig.add_subplot(111)
#ax.hist(ctrl_dff2,50,range=(-0.4,0.4))
ax.hist(ctrl_dff,50,range=(-0.4,0.4))
ax.set_xlabel("ctrl dFF")
ax.set_ylabel("n")

fig = plt.figure(4)
ax = fig.add_subplot(111)
ax.hist(ctrl_sd,50)
ax.set_xlabel("ctrl second derivative")
ax.set_ylabel("n")

plt.show()        
