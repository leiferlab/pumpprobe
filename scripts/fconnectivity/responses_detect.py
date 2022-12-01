import sys
import numpy as np
import wormdatamodel as wormdm
import wormbrain as wormb
import matplotlib.pyplot as plt
import os
import mistofrutta as mf
import pumpprobe as pp

folder = sys.argv[1]
if folder[-1]!="/": folder += "/"

tubatura = pp.Pipeline("responses_detect.py",folder=folder)
tubatura.open_logbook_f()

delta_t_pre = 30.0
ratio_method = 0.0
sig_type = "ratio"
ampl_thresh = 1.
ampl_min_time = 10.0
deriv_thresh = 1.#1.3
deriv_min_time = 2#3.5
nan_thresh = 0.3
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
smooth_mode = "sg"
smooth_n = 5
smooth_poly = 1
for s in sys.argv[1:]:
    sa = s.split(":")
    if sa[0] == "--ratio-method": ratio_method = float(sa[1])
    if sa[0] == "--signal": sig_type = sa[1]
    if sa[0] == "--ampl-thresh": ampl_thresh = float(sa[1])
    if sa[0] == "--deriv-thresh": deriv_thresh = float(sa[1])
    if sa[0] == "--ampl-min-time": ampl_min_time = float(sa[1])
    if sa[0] == "--deriv-min-time": deriv_min_time = float(sa[1])
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    if sa[0] == "--smooth-mode": smooth_mode = sa[1]
    if sa[0] == "--smooth-n": smooth_n = int(sa[1])
    if sa[0] == "--smooth-poly": smooth_poly = int(sa[1])
    

if not os.path.isdir(folder+"responses/"): os.mkdir(folder+"responses/")
if (matchless_nan_th is not None or matchless_nan_th_from_file) and sig_type != "green":
    raise ValueError("--matchless-nan-th can only be used with --signal:green")

try: 
    fconn_old = pp.Fconn.from_file(folder)
    if "-y" in sys.argv:
        save = True
    elif "-u" in sys.argv:
        save = False
        update = True
    else:
        risposta = input("Do you want to overwrite the existing fconn or update the responding neurons? (y/n/u)")[0] 
        save = risposta == "y"
        update = risposta == "u"
except: 
    save = True
    
if "--no-save" in sys.argv: save = False

settings_dict = {"delta_t_pre": delta_t_pre, "ratio_method": ratio_method, 
                 "sig_type": sig_type, 
                 "ampl_thresh": ampl_thresh, "ampl_min_time": ampl_min_time,
                 "deriv_thresh": deriv_thresh, "deriv_min_time": deriv_min_time,
                 "nan_thresh": nan_thresh, "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_from_file":matchless_nan_th_from_file,
                 "matchless_nan_th_added_only":matchless_nan_th_added_only,
                 "save": save}
                 
tubatura.log("",False)
tubatura.log("",False)
tubatura.log("## Detecting responses with the following configuration:",False)
tubatura.log("Command used: python "+" ".join(sys.argv),False)
tubatura.log(settings_dict.__repr__(),False)

# Get the recording object
rec = wormdm.data.recording(folder)
shift_vol = int(delta_t_pre/rec.Dt)
events = rec.get_events(shift_vol)

# Load the signal
if sig_type == "ratio":
    sig = wormdm.signal.Signal.from_signal_and_reference(folder,
                                                         method=ratio_method)
elif sig_type == "green":
    sig = wormdm.signal.Signal.from_file(
            folder,"green",
            matchless_nan_th=matchless_nan_th,
            matchless_nan_th_from_file=matchless_nan_th_from_file,
            matchless_nan_th_added_only=matchless_nan_th_added_only)
    sig.appl_photobl()
# Smooth and calculate the derivative of the signal (derivative needed for
# detection of responses)
sig.remove_spikes()
#sig.median_filter();sig.median_filter();sig.median_filter()
sig.smooth(n=smooth_n,i=None,poly=smooth_poly,mode=smooth_mode) 
tubatura.log("Smoothing with n="+str(smooth_n)+" before fconn")
sig.derivative = sig.get_derivative(sig.unsmoothed_data,11,1)
                            
# Make the zprojection of the reference volume
ref_index = sig.info['ref_index']
ref_vol = rec.get_vol(ref_index)
ref_zproj = np.sum(ref_vol,axis=0)[0]

# Cut the image for plotting
if not os.path.isfile(folder+'rectangle.txt'):
    rectangle = mf.geometry.draw.rectangle(ref_zproj)
    rect_coords = rectangle.getRectangle()
    np.savetxt(folder+"rectangle.txt",rect_coords)
else:
    rect_coords = np.loadtxt(folder+"rectangle.txt")
    
# Get the neurons coordinates of the reference volume and load the matches
# to determine what neuron was targeted
cervelli = wormb.Brains.from_file(folder)
ref_brain = cervelli(vol=ref_index)
cervelli.load_matches(folder)
labels = cervelli.get_labels(ref_index)

# Create functional connectome
fconn = pp.Fconn.from_objects(
                    rec,cervelli,sig,delta_t_pre,
                    deriv_thresh=deriv_thresh,deriv_min_time=deriv_min_time,
                    ampl_min_time=ampl_min_time,ampl_thresh=ampl_thresh,
                    nan_thresh=nan_thresh)
#print("Smoothing after the creation of the Fconn object")
#sig.smooth(n=39,i=None,poly=3,mode="sg")
for ie in np.arange(fconn.n_stim):   
    target_index_ref = fconn.stim_neurons[ie]
    target_label = str(target_index_ref)+":"+labels[target_index_ref]+"*"
    if target_index_ref==-3 and fconn.stim_neurons_compl_labels[ie] is not None:
        target_label = fconn.stim_neurons_compl_labels[ie]+"*"
    target_coords = events['optogenetics']['properties']['target'][ie]
    
    i0 = max(0,fconn.i0s[ie])
    i1 = fconn.i1s[ie]
    shift_vol = fconn.shift_vols[ie]
    time = (np.arange(i1-i0)-shift_vol)*rec.Dt
    
    sig_seg = sig.get_segment(i0,i1,shift_vol,baseline_mode="exp")
        
    selected = fconn.resp_neurons_by_stim[ie]
    if len(selected) == 0 and ie==0: continue
    
    alpha_min=0.1
    min_ampl = 0.
    max_ampl = 10./rec.Dt*5.
    rel_ampl = np.clip(fconn.resp_ampl_by_stim[ie],None,max_ampl)
    rel_ampl /= (max_ampl-min_ampl)
    alphas = alpha_min+rel_ampl*(1.-alpha_min)
    
    ######
    # PLOT
    ######
    
    overlay_label_dx = 0
    overlay_label_dy = 0
    
    if ie<fconn.n_stim-1: int_btw_stim = (fconn.i0s[ie+1]-i0)*rec.Dt
    else: int_btw_stim = None
    
    fig = plt.figure(1,figsize=(15,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.imshow(ref_zproj,aspect="auto")
    
    # Plot position of targeted neuron and raw target position
    if target_index_ref not in [-3,-2,-1]:
        ax2.plot(ref_brain[target_index_ref,2],ref_brain[target_index_ref,1],'xr')
    ax2.plot(target_coords[0],target_coords[1],'+',color="orange")
    
    if selected.shape[0] >0 and target_index_ref!=-2:
        for sel,alpha in zip(selected,alphas):
            if labels[sel]!="": lbl = str(sel)+":"+labels[sel]
            else: lbl = str(sel)
            if sel == target_index_ref: lw=3; lbl=target_label;
            else: lw=2;
            yplt = sig_seg[:,sel].copy()
            yplt[np.abs(yplt)>1e10] = np.nan
            l, = ax1.plot(time,yplt,label=lbl,lw=lw,alpha=alpha)
            if rect_coords[0,1]<ref_brain[sel,2]<rect_coords[1,1] and rect_coords[0,0]<ref_brain[sel,1]<rect_coords[1,0]:
                ax2.plot(ref_brain[sel,2],ref_brain[sel,1],'or',markersize=2)
                if labels[sel] != "": lbl = labels[sel]
                ax2.annotate(lbl,xy=(ref_brain[sel,2],ref_brain[sel,1]),
                                           xytext=(ref_brain[sel,2]+overlay_label_dx,
                                                   ref_brain[sel,1]+overlay_label_dy),
                                           color="r",
                                           fontsize=8)
    if target_index_ref not in selected and target_index_ref not in [-3,-2,-1]:
        sel = target_index_ref
        lw=3; lbl = target_label
        yplt = sig_seg[:,sel].copy()
        yplt[np.abs(yplt)>1e10] = np.nan
        ax1.plot(time,yplt,label=lbl,lw=lw)
        ax2.plot(ref_brain[sel,2],ref_brain[sel,1],'or',markersize=2)
        ax2.annotate(lbl,xy=(ref_brain[sel,2],ref_brain[sel,1]),
                                   xytext=(ref_brain[sel,2]+overlay_label_dx,
                                           ref_brain[sel,1]+overlay_label_dy),
                                   color="r",
                                   fontsize=8)
    elif target_index_ref not in selected and target_index_ref == -3:
        ax1.plot(0,0,label=target_label,c="k",ls="",marker="x")
            
        
    ax1.legend(loc=2)
        
        
    if int_btw_stim is not None:
        ax1.axvspan(int_btw_stim,time[-1],alpha=0.1,color="gray")
    ax1.axvline(0,c='k')
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("G/R (arb. u.)")
    ax1.set_title("Responses to stim. "+str(ie)+" by index of resp. neuron.")
    
    ax2.set_xlim(rect_coords[0,1],rect_coords[1,1])
    ax2.set_ylim(rect_coords[1,0],rect_coords[0,0])
    ax2.set_title("Responding neurons; x,+ location of stim.")
    
    plt.savefig(folder+"responses/stim_"+str(ie)+".png",bbox_inches="tight")
    fig.clf()

if save: 
    fconn.to_file(folder) 
elif update:
    fconn_old = fconn.from_file(folder)
    fconn_old.resp_neurons_by_stim = fconn.resp_neurons_by_stim
    fconn_old.resp_ampl_by_stim = fconn.resp_ampl_by_stim
    fconn_old.targeted_neuron_hit = fconn.targeted_neuron_hit
    fconn_old.nan_thresh = fconn.nan_thresh
    fconn_old.deriv_thresh = fconn.deriv_thresh
    fconn_old.ampl_thresh = fconn.ampl_thresh
    fconn_old.deriv_min_time = fconn.deriv_min_time
    fconn_old.ampl_min_time = fconn.ampl_min_time
    fconn_old.to_file(folder)
    
