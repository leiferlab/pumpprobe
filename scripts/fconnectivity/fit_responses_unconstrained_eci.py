import sys
import numpy as np
import wormdatamodel as wormdm
import wormbrain as wormb
import matplotlib.pyplot as plt
import os
import mistofrutta as mf
import pumpprobe as pp
from scipy.signal import savgol_coeffs

folder = sys.argv[1]
if folder[-1]!="/": folder += "/"
save = "--no-save" not in sys.argv
sig_green = "--signal:green" in sys.argv
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
for s in sys.argv[1:]:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])

if (matchless_nan_th is not None or matchless_nan_th_from_file) and not sig_green:
    raise ValueError("--matchless-nan-th can only be used with --signal:green")

if not os.path.isdir(folder+"responses/fits/"): os.mkdir(folder+"responses/fits/")

tubatura = pp.Pipeline("fit_responses_unconstrained_eci.py",folder=folder)
tubatura.open_logbook_f()
tubatura.log("",False)
tubatura.log("",False)
tubatura.log("## Fitting the responses with unconstrained ExponentialConvolutions.")
tubatura.log('Command used: python '+" ".join(sys.argv),False)

# Load the signal
if not sig_green:
    sig = wormdm.signal.Signal.from_signal_and_reference(folder)
else:
    tubatura.log("Using green signal.")
    sig = wormdm.signal.Signal.from_file(
                folder,"green",matchless_nan_th=matchless_nan_th,
                matchless_nan_th_from_file=matchless_nan_th_from_file)
    sig.appl_photobl()

# Smooth and calculate the derivative of the signal (derivative needed for
# detection of responses)
sig.remove_spikes()
#sig.median_filter()
n_smooth = 5
poly_smooth = 1
sig.smooth(n=n_smooth,i=None,poly=poly_smooth,mode="sg") 

# Get the neurons coordinates of the reference volume and load the matches
# to determine what neuron was targeted
cervelli = wormb.Brains.from_file(folder,ref_only=True)
labels = cervelli.get_labels(0)

# Create functional connectome
fconn = pp.Fconn.from_file(folder)
#shift_vol = fconn.shift_vol 

for ie in np.arange(fconn.n_stim): 
    i0 = max(0,fconn.i0s[ie])
    i1 = fconn.i1s[ie]
    shift_vol = fconn.shift_vols[ie]
    time = (np.arange(i1-i0)-shift_vol)*fconn.Dt
    
    stim = fconn.stim_neurons[ie]
    responding = fconn.resp_neurons_by_stim[ie]
    if stim not in responding:
        responding = np.append(responding,stim)
    responding_original = responding.copy()
    # Plot only the detected responses
    n_responding_original = len(responding_original)
    # but fit everything - nope
    #responding = np.arange(fconn.n_neurons)
    #n_responding = len(responding)
    responding = responding_original
    n_responding = n_responding_original
    
    nrows = max(1,int(np.sqrt(n_responding_original)))
    ncols = int(np.sqrt(n_responding_original))+2
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,10))
    #fig2, ax2 = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,10))
    #fig3, ax3 = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,10))
    for a in np.ravel(ax): a.set_xticks([]);a.set_yticks([])
    #for a in np.ravel(ax2): a.set_xticks([]);a.set_yticks([])
    #for a in np.ravel(ax3): a.set_xticks([]);a.set_yticks([])
    
    if nrows==1:
        ax = np.array([ax])
        #ax2 = np.array([ax2])
        #ax3 = np.array([ax3])
        
    for j in np.arange(n_responding):
        neu_j = responding[j]
        if neu_j<0: continue
        
        if ie<1:
            i1p = None
        elif ie<fconn.n_stim-2:
            inplus2 = neu_j in fconn.resp_neurons_by_stim[ie+2]
            inplus1 = neu_j in fconn.resp_neurons_by_stim[ie+1]
            i1p = None
            if inplus2: i1p = shift_vol+np.sum(fconn.next_stim_after_n_vol[ie:ie+2])
            if inplus1: i1p = shift_vol+fconn.next_stim_after_n_vol[ie]
        elif ie==fconn.n_stim-2:
            inplus1 = neu_j in fconn.resp_neurons_by_stim[ie+1]
            i1p = None
            if inplus1: i1p = shift_vol+fconn.next_stim_after_n_vol[ie]
        else:
            i1p = None
        
        x = time[shift_vol:i1p]
        
        # Determine the range for baseline subtraction. Keep the full shift_vol
        # interval if the neuron was not responding before. But shorten it
        # if the neuron was responding to the previous stimulation. This latter
        # case is more sensitive to the noise, but avoids systematic wrong
        # baselines due to ongoing dynamics in the shift_vol segment.
        if ie>0:
            if neu_j in fconn.resp_neurons_by_stim[ie-1]:
                y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                    baseline_mode="constant",
                                    baseline_range=[shift_vol-4,shift_vol],
                                    normalize="none")[:,neu_j]
            else:
                y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                    baseline_mode="constant",
                                    normalize="none")[:,neu_j]
        else:
            y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                    baseline_mode="constant",
                                    normalize="none")[:,neu_j]
        
        sm = savgol_coeffs(n_smooth,poly_smooth)
        y_sm = np.convolve(y,sm,mode="same")
        if np.all(np.isinf(y)) or np.all(np.isnan(y)): continue
        y[np.isinf(y)] = y[np.where(np.isinf(y))[0]-1]
        y_plt = y
        y = y_plt[shift_vol:i1p]
        
        loc_std = sig.get_loc_std(y,4)
        
        fconn.clear_fit_results(stim=ie,neu=neu_j)
        
        rms_calc_lim = min(int(30/fconn.Dt),len(x))
        
        n_hops_min = 2
        params,rms = pp.Fconn.fit_eci(x,y,n_min=n_hops_min,n_max=8,
                                      routine="least_squares",method="trf",
                                      rms_limits=[0,rms_calc_lim],
                                      auto_stop=True,rms_tol=1e-2)
        if params is None: 
            tubatura.log("params is None. stim "+str(ie)+" neuron "+str(neu_j))
            params_dict = {"params": [0,1], 
                       "n_branches": 1, 
                       "n_branch_params": [2]}
            fconn.fit_params_unc[ie][neu_j] = params_dict
            continue
        
        rms_argsort = np.argsort(rms)
        winner = rms_argsort[0]
        advantage = (rms[winner]-rms[rms_argsort[1]])/rms[rms_argsort[1]]
        
        n_hops_winner = n_hops_min+winner
        params_dict = {"params": params[winner], 
                       "n_branches": 1, 
                       "n_branch_params": [n_hops_winner+1]}
        fconn.fit_params_unc[ie][neu_j] = params_dict
        
        if neu_j in responding_original:
            # Plot only if the response was detected
            j_plot = np.where(responding_original==neu_j)[0][0]
            ax_r = j_plot//ncols
            ax_c = j_plot%ncols
            
            lbl = str(neu_j)+":"+labels[neu_j]#
            lw = 1
            if neu_j==stim: 
                lbl+="*"
                lw = 3
                
            ax[ax_r,ax_c].plot(time,y_plt,label=lbl,lw=lw)
            ax[ax_r,ax_c].plot(time,y_sm,label=lbl,lw=lw)
            
            for q in np.arange(len(params)):
                p = q + n_hops_min
                fit_y = pp.Fconn.eci(x,params[q])
                fit_ls = "-" if q==winner else ":"
                if q==winner: 
                    fit_lbl = "ec"+str(p)+" "+str(np.around(advantage,3))

                    ax[ax_r,ax_c].plot(x,fit_y,label=fit_lbl,lw=2,ls=fit_ls)
                    #ax2[ax_r,ax_c].plot(x,y-fit_y,label=fit_lbl,lw=2,ls=fit_ls)
                else:
                    ax[ax_r,ax_c].plot(x,fit_y,lw=2,ls=fit_ls)
                    #ax2[ax_r,ax_c].plot(x,y-fit_y,lw=2,ls=fit_ls)
            
            ax[ax_r,ax_c].set_xlim(time[0],time[-1])
            ax[ax_r,ax_c].axvline(0,c="k",alpha=0.5)
            ax[ax_r,ax_c].axhline(0,c="k")
            ax[ax_r,ax_c].axvline(fconn.next_stim_after_n_vol[ie]*fconn.Dt,c="k",alpha=0.5)
            ax[ax_r,ax_c].legend(fontsize=7)
            
            #ax2[ax_r,ax_c].axhline(0,c="k")
            #ax2[ax_r,ax_c].legend()
        
    if sig_green: sig_type="g_"
    else: sig_type=""
    plt.figure(1)
    plt.tight_layout()
    plt.savefig(folder+"responses/fits/"+sig_type+"eci_stim"+str(ie)+".png",bbox_inches="tight")
    #plt.figure(2)
    #plt.tight_layout()
    #plt.savefig(folder+"responses/fits/res_"+sig_type+"eci_stim"+str(ie)+".png",bbox_inches="tight")
    plt.close(fig=fig);#plt.close(fig=fig2);#plt.close(fig=fig3)

if save: fconn.to_file(folder)
