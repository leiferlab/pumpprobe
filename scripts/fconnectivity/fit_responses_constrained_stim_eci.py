import sys
import numpy as np
import wormdatamodel as wormdm
import wormbrain as wormb
import matplotlib.pyplot as plt
import os
import mistofrutta as mf
import pumpprobe as pp

plot = True

folder = sys.argv[1]
if folder[-1]!="/": folder += "/"
save = "--no-save" not in sys.argv
sig_green = "--signal:green" in sys.argv
skip_if_not_manually_located = "--skip-if-not-manually-located" in sys.argv
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
for s in sys.argv[1:]:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    
if (matchless_nan_th is not None or matchless_nan_th_from_file) and not sig_green:
    raise ValueError("--matchless-nan-th can only be used with --signal:green")

if not os.path.isdir(folder+"responses/fits/"): os.mkdir(folder+"responses/fits/")

tubatura = pp.Pipeline("fit_responses_constrained_stim_eci.py",folder=folder)
tubatura.open_logbook_f()
tubatura.log("",False)
tubatura.log("",False)
tubatura.log("## Fitting the responses with stim_constrained ExponentialConvolutions.")
tubatura.log('Command used: python '+" ".join(sys.argv),False)

# Load the signal
if not sig_green:
    sig = wormdm.signal.Signal.from_signal_and_reference(folder)
else:
    tubatura.log("Using green signal.")
    sig = wormdm.signal.Signal.from_file(
                folder,"green",
                matchless_nan_th=matchless_nan_th,
                matchless_nan_th_from_file=matchless_nan_th_from_file,
                matchless_nan_th_added_only=matchless_nan_th_added_only)
    sig.appl_photobl()
# Smooth and calculate the derivative of the signal (derivative needed for
# detection of responses)
sig.remove_spikes()
#sig.median_filter()
#sig.smooth(n=127,i=None,poly=7,mode="sg")

# Get the neurons coordinates of the reference volume and load the matches
# to determine what neuron was targeted
cervelli = wormb.Brains.from_file(folder,ref_only=True)
labels = cervelli.get_labels(0)

# Create functional connectome
fconn = pp.Fconn.from_file(folder)
#shift_vol = fconn.shift_vol

# Check that the targets have been manually located, and, if not, ask for 
# confirmation.
if not fconn.manually_located_present:
    tubatura.log("Targets have not been manually located/confirmed.")
    if not skip_if_not_manually_located:
        cont = input("Continue? (y/n)") == "y"
        if not cont: quit() 
    else:
        print("\tSkipping this dataset.")
        quit()
        
tubatura.log("Fitting with n_branches_max = 2")
for ie in np.arange(fconn.n_stim): 
    i0 = max(0,fconn.i0s[ie])
    i1 = fconn.i1s[ie]
    shift_vol = fconn.shift_vols[ie]
    time = (np.arange(i1-i0)-shift_vol)*fconn.Dt
    
    # Get the indices of the stimulated neuron and of the responding neurons.
    stim = fconn.stim_neurons[ie]
    responding_original = fconn.resp_neurons_by_stim[ie]
    n_responding_original = len(responding_original)
    # but fit everything - nope
    #responding = np.arange(fconn.n_neurons)
    #n_responding = len(responding)
    responding = responding_original
    n_responding = n_responding_original
    
    
    # Get the unconstrained parameters to build a cleaned-up version of the
    # stimulated neuron's activity.
    stim_unc_par_dict = fconn.fit_params_unc[ie][stim]
    
    #########################################
    # CASES IN WHICH TO SKIP THIS STIMULATION
    #########################################
    # If the targeting has failed
    if stim == -2:
        tubatura.log("skipping "+str(ie)+" failed target");continue
    # If no response was detected in the stimulated neuron, because you'd risk
    # finding weird kernels just because there is no activity as input.
    if stim not in responding_original:
        tubatura.log("skipping "+str(ie)+" stimulated neuron did not respond");continue
    # If the fit of the stimulated neuron's activity did not converge.    
    if stim_unc_par_dict["n_branches"]==0:
        tubatura.log("skipping "+str(ie)+" fit absent");continue
    
    stim_unc_par = fconn.get_irrarray_from_params(stim_unc_par_dict)
    
    ###############
    # PREPARE PLOTS
    ###############
        
    nrows = max(1,int(np.sqrt(n_responding_original)))
    ncols = int(np.sqrt(n_responding_original))+2
    if plot:
        print("plotting")
        try:
            fig.clear()
        except:
            pass
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15,10))
        for a in np.ravel(ax): a.set_xticks([]);a.set_yticks([])
        if nrows==1: ax = np.array([ax])
        
    for j in np.arange(n_responding):
        neu_j = responding[j]
        if neu_j==stim: continue
        
        # Skip the following checks on i1 and simply fit on the time axis
        # before the next stimulus. This also avoids fits of the next response.
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
        #i1p = shift_vol+fconn.next_stim_after_n_vol[ie]
        if ie == fconn.n_stim-1: i1p = None
                
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
                                    baseline_range=[shift_vol-4,shift_vol])[:,neu_j]#, FIXME FIXME FIXME
                                    #normalize="none")[:,neu_j]
            else:
                y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                    baseline_mode="constant")[:,neu_j]#, FIXME FIXME FIXME
                                    #normalize="none")[:,neu_j]
        else:
            y = sig.get_segment(i0,i1,shift_vol,unsmoothed_data=True,
                                    baseline_mode="constant")[:,neu_j]#, FIXME FIXME FIXME
                                    #normalize="none")[:,neu_j]
        
        #y = sig.get_segment(i0,i1,shift_vol)[:,neu_j]
        if np.all(np.isinf(y)) or np.all(np.isnan(y)): continue
        y[np.isinf(y)] = y[np.where(np.isinf(y))[0]-1]
        y_plt = y
        y = y_plt[shift_vol:i1p]
        #if y.shape[0] == 0: continue
        
        stim_y = pp.Fconn.eci(x,stim_unc_par)
                
        loc_std = sig.get_loc_std(y,4)
        
        fconn.clear_fit_results(stim=ie,neu=neu_j,mode="constrained")
        
        rms_calc_lim = min(int(30/fconn.Dt),len(x))
        
        n_hops_min = 2
        params_, n_branch_params, _ = fconn.fit_eci_branching(
                          x,y,stim_y,dt=fconn.Dt,
                          n_hops_min=1,n_hops_max=3,
                          n_branches_max=2,#3,
                          rms_limits=[None,None],auto_stop=True,rms_tol=1e-2,
                          method="trf",routine="least_squares")
        
        if params_ is None: 
            tubatura.log("constrained params is None. stim "+str(ie)+" neuron "+str(neu_j))
            params_dict = {"params": [0,1], 
                       "n_branches": 1, 
                       "n_branch_params": [2]}
            fconn.fit_params[ie][neu_j] = params_dict
            continue
        
        params_dict = {"params": np.array(params_), 
                       "n_branches": len(n_branch_params), 
                       "n_branch_params": n_branch_params}
        fconn.fit_params[ie][neu_j] = params_dict
        
        if neu_j in responding_original and plot:
            #print("plotting")
            # Plot only for detected responses
            j_plot = np.where(responding_original==neu_j)[0][0]
            ax_r = j_plot//ncols
            ax_c = j_plot%ncols
            
            #lbl = str(neu_j)
            lbl = str(neu_j)+":"+labels[neu_j]
            lw = 1
            if neu_j==stim: 
                lbl+="*"
                lw = 3
                
            ax[ax_r,ax_c].plot(time,y_plt,label=lbl,lw=lw)
            
            params = fconn.get_irrarray_from_params(params_dict)
            rf = pp.Fconn.eci(x,params)
            fit_y = pp.convolution(stim_y,rf,fconn.Dt,8)
            fit_ls = "-"
            fit_lbl = "|".join([str(nbp-1) for nbp in n_branch_params])
            
            rf_plt = rf
            rf_plt /= np.max(np.abs(rf_plt))/np.max(np.abs(fit_y))
            stim_y_plt = stim_y/np.sum(stim_y)*np.abs(np.sum(y))
            
            ax[ax_r,ax_c].plot(x,fit_y,label=fit_lbl,lw=2,ls=fit_ls)
            ax[ax_r,ax_c].plot(x,rf_plt,label="rf",lw=2,c="k")
            ax[ax_r,ax_c].plot(x,stim_y_plt,label="st",lw=2,c="yellow",alpha=0.5)
            
            ax[ax_r,ax_c].set_xlim(time[0],time[-1])
            ax[ax_r,ax_c].set_ylim(min(y_plt),max(y_plt))
            ax[ax_r,ax_c].axvline(0,c="k",alpha=0.5)
            ax[ax_r,ax_c].axvline(fconn.next_stim_after_n_vol[ie]*fconn.Dt,c="k",alpha=0.5)
            ax[ax_r,ax_c].legend()
        
    if plot:
        if sig_green: sig_type="g_"
        else: sig_type=""
        plt.figure(1)
        plt.tight_layout()
        plt.savefig(folder+"responses/fits/"+sig_type+"eci_con_stim"+str(ie)+".png",bbox_inches="tight")
        plt.close(fig=fig)

if save: fconn.to_file(folder)
