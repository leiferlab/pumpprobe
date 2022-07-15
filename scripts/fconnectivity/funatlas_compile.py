import numpy as np, matplotlib.pyplot as plt, sys, re, os
import pumpprobe as pp, wormdatamodel as wormdm, wormbrain as wormb
from scipy.signal import savgol_coeffs
from scipy.stats import combine_pvalues
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)

# List of datasets
fname = "/projects/LEIFER/francesco/funatlas_list.txt"

# Parse inputs
i = j = ""
all_pairs = "--all-pairs" in sys.argv
use_kernels = "--use-kernels" in sys.argv
drop_saturation_branches = "--drop-saturation-branches" in sys.argv
ds_tags = None
ds_exclude_tags = None
signal = "green"
req_auto_response = "--req-auto-response" in sys.argv or use_kernels
enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
inclall_occ = "--inclall-occ" in sys.argv
nan_th = 1 # To use with inclall-occ
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
ylabel = None
ylim = [-1.1,1.1]
yticks = None
figsize = None
two_min_occ = "--two-min-occ" in sys.argv
normalize = ""
smooth_mode = "sg"
smooth_n = 13
smooth_poly = 1
invert_title = "--invert-title" in sys.argv
legend = "--no-legend" not in sys.argv
single_color = "--single-color" in sys.argv
plot_average = "--plot-average" in sys.argv
stamp_plot = "--no-stamp" not in sys.argv
alpha_p = "--alpha-p" in sys.argv
verbose = "--no-verbose" not in sys.argv
if not verbose: print("Suppressing prints from Fconn.")
dst = "/projects/LEIFER/francesco/funatlas/plots/"
fmt = "png"
dpi = 150
user = "Francesco"
exclude = []
for s in sys.argv:
    sa = s.split(":")
    if sa[0] in ["-j","--j"] : jid = sa[1]
    if sa[0] in ["-i","--i"]: iid = sa[1]
    if sa[0] == "--signal": signal = sa[1]
    #if sa[0] == "--leq-rise-time": leq_rise_time = float(sa[1])
    #if sa[0] == "--geq-rise-time": geq_rise_time = float(sa[1])
    #if sa[0] == "--leq-decay-time": leq_decay_time = float(sa[1])
    #if sa[0] == "--geq-decay-time": geq_decay_time = float(sa[1])
    if sa[0] == "--ylabel": ylabel = sa[1]
    if sa[0] == "--ylim": 
        if sa[1] in ["none","None"]:
            ylim = None
        else:
            ylim = [float(sb) for sb in sa[1].split(",")]
    if sa[0] == "--yticks": yticks = [float(sb) for sb in sa[1].split(",")]
    if sa[0] == "--figsize": figsize = [int(sb) for sb in sa[1].split(",")]
    if sa[0] == "--normalize": normalize = sa[1]
    if sa[0] == "--smooth-mode": smooth_mode = sa[1]
    if sa[0] == "--smooth-n": smooth_n = int(sa[1])
    if sa[0] == "--smooth-poly": smooth_poly = int(sa[1])
    if sa[0] == "--nan-th": nan_th = float(sa[1])
    if sa[0] == "--list": fname = sa[1]
    if sa[0] == "--dst": dst = sa[1] if sa[1][-1]=="/" else sa[1]+"/"
    if sa[0] == "--fmt": fmt=sa[1]
    if sa[0] == "--dpi": dpi=int(sa[1])
    if sa[0] == "--user": user = sa[1]
    if sa[0] == "--ds-tags": ds_tags = " ".join(sa[1].split("-"))
    if sa[0] == "--ds-exclude-tags": ds_exclude_tags = " ".join(sa[1].split("-"))
    if sa[0] == "--exclude": #To exclude noisy traces
        sb = sa[1].split("-")
        for i in np.arange(len(sb)):
            sc = sb[i].split(",")
            if len(sc)==1: ex = {"ds":int(sc[0])}
            elif len(sc)==2: ex = {"ds":int(sc[0]),"stim":int(sc[1])}
            elif len(sc)==3: ex = {"ds":int(sc[0]),"stim":int(sc[1]),
                                   "resp_neu_i":int(sc[2])}
            exclude.append(ex)
if not all_pairs and (iid=="" or jid==""): print("Select i and j.");quit()

if ylabel is None: ylabel = signal

if user in ["Sophie","sophie"]:
    dst = "/projects/LEIFER/Sophie/funatlas/plots/"
    fname = "/projects/LEIFER/Sophie/funatlas/funatlas_list.txt"
elif user in ["Andy","andy"]:
    dst = "/projects/LEIFER/andy/funatlas/plots/"
    fname = "/projects/LEIFER/andy/funatlas/funatlas_list.txt"
    
if not os.path.isdir(dst):
    os.mkdir(dst)
    
# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file}

# Build Funatlas object
funatlas = pp.Funatlas.from_datasets(
                fname,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal=signal,signal_kwargs=signal_kwargs,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck,
                verbose=verbose)

# Save the ds list used                
funatlas.ds_to_file(dst,fname="aaa_funatlas_ds_list_used.txt")

# Occurence matrix. occ1[i,j] is the number of occurrences of the response of i
# following a stimulation of j. occ2[i,j] is a dictionary containing details
# to extract the activities from the signal objects.
occ1, occ2 = funatlas.get_occurrence_matrix(
                    req_auto_response=req_auto_response,inclall=inclall_occ)
                    
                    
if alpha_p:
    # Multiply the alpha also by the p values
    no_stim_folders = [
                  "/projects/LEIFER/Sophie/NewRecordings/20220214/pumpprobe_20220214_171348/",
                  "/projects/LEIFER/Sophie/NewRecordings/20220215/pumpprobe_20220215_112405/",
                  "/projects/LEIFER/Sophie/NewRecordings/20220216/pumpprobe_20220216_161637/"]
    # Load Funatlas for spontaneous activity                                 
    funa_ctrl = pp.Funatlas.from_datasets(no_stim_folders, merge_bilateral=True,
                                             signal="green")
    xdff,xsd,cdf = funa_ctrl.get_ctrl_cdf()
    
    n_ds = len(funatlas.ds_list)
    sder = np.empty(n_ds,dtype=object)
    for a in sder: a = []
    for ds in np.arange(n_ds):
        sderker = savgol_coeffs(13, 2, deriv=2, delta=funatlas.fconn[ds].Dt)
        sder_ = np.zeros_like(funatlas.sig[ds].data)            
        for k in np.arange(funatlas.sig[ds].data.shape[1]):
            sder_[:,k] = np.convolve(sderker,funatlas.sig[ds].data[:,k],
                                     mode="same")
        sder[ds] = sder_
        
     
dff = False            
if dff:   
    print("FILTERING BY DFF")
    timedff = np.linspace(0,60,120)
    dFF = funatlas.get_max_deltaFoverF(occ2,timedff)
    occ2,_ = funatlas.filter_occ2(occ2,dFF,leq=0.1)

# If occ2 needs to be filtered
occ1,occ2 = funatlas.filter_occ12_from_sysargv(occ2,sys.argv)

if all_pairs: 
    iids = jids = funatlas.neuron_ids
else: 
    if jid == "*":
        jids = funatlas.neuron_ids
    else:
        jids = jid.split(",")
    if iid=="j":
        iids = jids.copy()
    elif iid=="*":
        iids = funatlas.neuron_ids
    else:
        iids = iid.split(",")
    
    #iids = [iid]; jids = [jid]

if figsize is None: fig = plt.figure(1)
else: fig = plt.figure(1,figsize=figsize)
for iid in iids:
    for jid in jids:
        # Convert the requested IDs to atlas-indices.
        i,j = funatlas.ids_to_i([iid,jid])
        if i<0: print(iid,"not found. Check approximations.")
        if j<0: print(jid,"not found. Check approximations.")
        
        if (two_min_occ and occ1[i,j]<2) or occ1[i,j]==0: continue
        
        # Plot the responses.
        fig.clear()
        ax1 = fig.add_subplot(111)
        ls = []
        # Get the necessary information from occ2.
        ys = []
        times = []
        confidences = []
        for occ in occ2[i,j]:
            ds = occ["ds"]
            ie = occ["stim"]
            neu_i = occ["resp_neu_i"]
            neu_j = funatlas.fconn[ds].stim_neurons[ie]
            
            # Iterate over the exclude array to see if this trace should be
            # excluded.
            skip = False
            for iex in np.arange(len(exclude)):
                ex = exclude[iex]
                if (len(ex.keys())==1 and ds == ex["ds"]) or \
                   (len(ex.keys())==2 and ds == ex["ds"] and ie == ex["stim"]) or\
                   (len(ex.keys())==3 and ds == ex["ds"] and ie == ex["stim"] and neu_i == ex["resp_neu_i"]):
                       skip = True
            if skip: continue
            
            # Build the time axis
            i0 = funatlas.fconn[ds].i0s[ie]
            i1 = funatlas.fconn[ds].i1s[ie]
            Dt = funatlas.fconn[ds].Dt
            shift_vol = funatlas.fconn[ds].shift_vols[ie]
            if not use_kernels:
                time = (np.arange(i1-i0)-shift_vol)*Dt
            else:
                time = np.linspace(0,20,100)
            
            if not use_kernels: # get the signal trace
                norm_range = (None,shift_vol+int(40./Dt))
                y = funatlas.sig[ds].get_segment(i0,i1,shift_vol,
                                                 baseline_range=(shift_vol-10,None),
                                                 normalize=normalize,
                                                 norm_range=norm_range)[:,neu_i]
                if nan_th<1:
                    nany = funatlas.sig[ds].get_segment_nan_mask(i0,i1)[:,neu_i]
                    if np.sum(nany)>nan_th*len(nany): continue
            else:
                y_ec = funatlas.fconn[ds].get_kernel_ec(ie,neu_i)
                if y_ec is None: 
                    continue
                else: 
                    if drop_saturation_branches:
                        y_ec = y_ec.drop_saturation_branches()
                    y=y_ec.eval(time)
                
                if normalize=="max_abs":
                    y /= np.max(np.abs(y))
            
            if not plot_average:    
                # Use the labeling confidence to set the alpha of the line.
                conf_i = funatlas.labels_confidences[ds][neu_i]
                conf_j = funatlas.labels_confidences[ds][neu_j]
                if conf_i == -1 and conf_j == -1: conf = 0.25
                elif conf_i == -1: conf = conf_j
                elif conf_j == -1: conf = conf_i
                else: conf = conf_i*conf_j
                alpha = conf#do this later np.clip(conf,0.25,None)
                #alpha = (alpha-0.25)/0.75
                lw = 1
            else:
                alpha = 0.7
                lw = 0.5
                ys.append(y)
                times.append(time)
                confidences.append((np.clip(funatlas.labels_confidences[ds][i],0.5,None)-0.5)*2)
                
            if alpha_p:
                # Multiply alpha by ~1-p value
                act = []
                sd = []
                dff__j,_,sd__j = funatlas.get_significance_features(
                                                funatlas.sig[ds],
                                                neu_i,i0,i1,shift_vol,
                                                funatlas.fconn[ds].Dt,
                                                nan_th=0.3)
                
                    
                dff__i,_,sd__i = funatlas.get_significance_features(
                                                funatlas.sig[ds],
                                                neu_i,i0,i1,shift_vol,
                                                funatlas.fconn[ds].Dt,
                                                nan_th=0.3)
                
                if dff__j is not None and dff__i is not None:
                    act.append(dff__j)
                    sd.append(sd__j)
                    act.append(dff__i)
                    sd.append(sd__i)
                    '''activity = funatlas.sig[ds].get_segment(
                                            i0,i1,shift_vol,
                                            normalize="")[:,np.array([neu_j,neu_i])]
                    baseline = np.average(activity[:shift_vol],axis=0)
                    pre = baseline
                    pre = funatlas.sig[ds].get_loc_std(activity[:shift_vol],8)
                    act = np.average(activity[shift_vol:]-baseline,axis=0)/pre
                                            
                    sd_ = sder[ds][i0+shift_vol-5:i0+shift_vol+11,np.array([neu_j,neu_i])]
                    sd = np.sum(sd_,axis=0)'''
                                        
                    p_a_ = funatlas.get_individual_p(act,xdff,cdf)
                    p_b_ = funatlas.get_individual_p(sd,xsd,cdf)
                    p_a_[p_a_==0] = 1e-6
                    p_b_[p_b_==0] = 1e-6
                                   
                    _,p_j = combine_pvalues([p_a_[0],p_b_[0]],method="fisher")
                    _,p_i = combine_pvalues([p_a_[1],p_b_[1]],method="fisher")
                    #p_j = max(p_a_[0],p_b_[0])
                    #p_i = max(p_a_[1],p_b_[1])
                    
                    w_j = 1-p_j#max((1-p_j-0.99)/0.99,0)
                    w_i = 1-p_i#max((1-p_i-0.99)/0.99,0)
                    
                    w_ = min(w_i,w_j)
                    #w_ = max(w_-0.8,0)/0.2
                    #if iid=="IL1V_": print(w_i,w_j,w_)
                    alpha = w_
                else:
                    alpha = 0.0
            
            lbl = str(ds)+","+str(ie)+","+str(neu_i)
            if alpha_p: lbl += ","+str(np.around(w_,4))
            
            alpha = max((alpha-0.4)/0.6,0)
            if not single_color:
                l, = ax1.plot(time,y,
                              label=lbl,
                              alpha=alpha,lw=lw)
            else:
                l, = ax1.plot(time,y,
                              label=lbl,
                              alpha=alpha,lw=lw,c="#1f77b4")
            ls.append(l)
        
        if plot_average:
            t_min = np.max([min(time) for time in times])
            t_max = np.min([max(time) for time in times])
            time_avg = np.linspace(t_min,t_max,int((t_max-t_min)/0.5))
            y_avg = np.zeros_like(time_avg)
            ys_interp = np.zeros((len(ys)+1,len(y_avg)))
            ys_interp[0] = time_avg
            for iy in np.arange(len(ys)):
                y_interp = np.interp(time_avg,times[iy],ys[iy])
                y_avg += y_interp*confidences[iy]
                ys_interp[iy+1] = y_interp
            y_avg /= np.sum(confidences)#len(ys)
            
            np.save(dst+jid+"->"+iid+"_ys_interp",ys_interp)
            np.save(dst+jid+"->"+iid+"_confidences",np.array(confidences))
            
            c = "#1f77b4" if single_color else "k"
            ax1.plot(time_avg,y_avg,lw=2,c=c)
        
        ax1.axvline(0,c="k",alpha=0.5)
        if ylim is not None:
            ax1.set_ylim(ylim[0],ylim[1])
        if yticks is not None:
            ax1.set_yticks(yticks)
        ax1.set_xlabel("Time (s)",fontsize=18)
        ax1.set_ylabel(ylabel,fontsize=18)
        if not invert_title:
            title = iid+r"$\leftarrow$"+jid
        else:
            title = jid+r"$\rightarrow$"+iid
        if legend: title += " (ds,ie,i)"
        ax1.set_title(title,fontsize=18)
        if legend: 
            loc = 2 if not use_kernels else 1
            ax1.legend(loc=loc)
        if stamp_plot:
            stamp = " ".join(sys.argv)
            stamp=re.sub("(.{100})", "\\1\n", stamp, 0, re.DOTALL)
            pp.provstamp(ax1,-.1,-.1,stamp,6)
        fig.tight_layout()
        fig.savefig(dst+jid+"->"+iid+"."+fmt,dpi=dpi)
