import numpy as np, matplotlib.pyplot as plt, sys, re, matplotlib.gridspec as gridspec
import pumpprobe as pp, wormdatamodel as wormdm

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

relative = "--relative" in sys.argv
jid = "RID"
vmax = 1

matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
for s in sys.argv:
    sa = s.split(":")
    if sa[0] in ["-i","--i"] : iid=sa[1]
    if sa[0] in ["-j","--j"] : jid=sa[1]
    if sa[0] == "--vmax": vmax=float(sa[1])
    if sa[0] == "--matchless-nan-th": matchless_nan_th=float(sa[1])

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True, 
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

funa_wt = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=False,merge_dorsoventral=False,
                merge_numbered=False,merge_AWC=False,
                ds_exclude_tags="mutant",
                signal_kwargs=signal_kwargs)
                
funa_unc31 = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=False,merge_dorsoventral=False,
                merge_numbered=False,merge_AWC=False,
                ds_tags="unc31",
                signal_kwargs=signal_kwargs)
                
_, occ2_wt = funa_wt.get_occurrence_matrix(inclall=True)
_, occ2_unc31 = funa_unc31.get_occurrence_matrix(inclall=True)

ai_RID = funa_wt.ids_to_i(jid)

t = np.linspace(-10,30,120)
t0 = np.argmin(np.abs(t-0))

fig = plt.figure(1,figsize=(14,3))
spec = gridspec.GridSpec(ncols=1, nrows=4, figure=fig)
ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[1:,0])
fig.subplots_adjust(hspace=0.05)

xticks = []
xticklabels = []

n = 0
dn = 1.0
d = 0.2
bar_width = 0.2

dff_th = 0.1#0.1

for ai_i in np.arange(len(occ2_wt[:,ai_RID])):
    if ai_i == ai_RID: continue
    if len(occ2_wt[ai_i,ai_RID])==0 or len(occ2_unc31[ai_i,ai_RID])==0:
        continue
        
    resp_wt = []
    resp_unc31 = []

    avg_wt = []
    avg_unc31 = []

    for io in np.arange(len(occ2_wt[ai_i,ai_RID])):
        o = occ2_wt[ai_i,ai_RID][io]
        ds = o["ds"]
        ie = o["stim"]
        i = o["resp_neu_i"]
        
        time,time2,i0,i1 = funa_wt.fconn[ds].get_time_axis(ie,True)
        shift_vol = funa_wt.fconn[ds].shift_vols[ie]
        
        seg = funa_wt.sig[ds].get_segment(
                                    i0,i1,shift_vol,
                                    baseline=False,normalize="")[:,i]
        nan = funa_wt.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
                                    
        baseline = np.nanmean(seg[shift_vol//2:shift_vol])
        y = (seg-baseline)
        y = y/baseline
            
        if (ds==41 and ie==50 and i == 93): #extremely noisy segment that is not smoothed for some reason 
            alt_y = funa_wt.sig[ds].get_smoothed(13, i=93, poly=1, mode="sg_causal")
            alt_y = alt_y[i0:i1]
            baseline = np.nanmean(alt_y[:shift_vol])#shift_vol//2:
            y = alt_y-baseline
            y = y/baseline
        
        if np.sum(nan)<0.3*len(nan) :        
            avg_wt.append(np.nanmean(y[shift_vol:shift_vol+40]))
            resp_wt.append(np.interp(t,time,y))
            
    avg_wt = np.array(avg_wt)
    resp_wt = np.array(resp_wt)
            
    for io in np.arange(len(occ2_unc31[ai_i,ai_RID])):
        o = occ2_unc31[ai_i,ai_RID][io]
        ds = o["ds"]
        ie = o["stim"]
        i = o["resp_neu_i"]
        
        time,time2,i0,i1 = funa_unc31.fconn[ds].get_time_axis(ie,True)
        shift_vol = funa_unc31.fconn[ds].shift_vols[ie]
        
        seg = funa_unc31.sig[ds].get_segment(
                                    i0,i1,shift_vol,
                                    baseline=False,normalize="")[:,i]
        nan = funa_unc31.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
                                    
        baseline = np.nanmean(seg[:shift_vol])#shift_vol//2:
        y = (seg-baseline)
        y = y/baseline
        
        if np.sum(nan)<0.3*len(nan):
            avg_unc31.append(np.nanmean(y[shift_vol:shift_vol+40]))
            resp_unc31.append(np.interp(t,time,y))
    
    avg_unc31 = np.array(avg_unc31)        
    resp_unc31 = np.array(resp_unc31)
    
    if np.abs(np.average(avg_wt))<dff_th and np.abs(np.average(avg_unc31))<dff_th:
        continue
    
    if np.isnan(np.nanmean(avg_wt)) or np.isnan(np.nanmean(avg_unc31)):
        continue

    ax2.bar(n*dn,np.nanmean(avg_wt),color="C0",width=bar_width,alpha=0.6)
    ax2.bar(n*dn+d,np.nanmean(avg_unc31),color="C1",width=bar_width,alpha=0.6)
    
    ax1.scatter(n*dn+np.random.random(len(avg_wt))*bar_width - bar_width/2, avg_wt, color="C0",s=0.5)
    ax1.scatter(n*dn+d+np.random.random(len(avg_unc31))*bar_width - bar_width/2, avg_unc31, color="C1",s=0.5)
    
    ax2.scatter(n*dn+np.random.random(len(avg_wt))*bar_width - bar_width/2, avg_wt, color="C0",s=0.5)
    ax2.scatter(n*dn+d+np.random.random(len(avg_unc31))*bar_width - bar_width/2, avg_unc31, color="C1",s=0.5)
    
    
    xticks.append(n*dn+d/2)
    xticklabels.append(funa_wt.neuron_ids[ai_i])
    
    n += 1
    
ax2.axhline(0,color="k",lw=1)
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels,fontsize=6)
ax2.set_xlim(-1,n)
ax2.set_ylim(None,1.18)
ax2.set_ylabel(r"$\Delta F/F$")
ax2.spines.top.set_visible(False)
ax2.spines.right.set_visible(False)
ax2.xaxis.tick_bottom()

ax1.set_ylim(2,None)
ax1.set_yscale("log")
ax1.set_yticks([2,3,4])
ax1.set_yticklabels(["2","3","4"])
ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)
ax1.tick_params(labeltop=False,bottom=False,top=False)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)

#fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figSn_RID/RID_bar_plot.pdf",dpi=300,bbox_inches="tight")

plt.show()
