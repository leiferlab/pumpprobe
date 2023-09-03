import numpy as np, matplotlib.pyplot as plt, sys, re
import pumpprobe as pp, wormdatamodel as wormdm

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

#plt.rcParams["svg.image_inline"]=False#("svg",image_noscale = False)

inclall = "--inclall-occ" in sys.argv
sort_t = "--sort-t" in sys.argv
sort_min = "--sort-min" in sys.argv
sort_max = "--sort-max" in sys.argv
sort_avg = "--sort-avg" in sys.argv
relative = "--relative" in sys.argv
merge_bilateral = "--merge-bilateral" in sys.argv
iid = ""
jid = "RID"
vmax = 1
matchless_nan_th = 0.0#1.
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
                ds_list,merge_bilateral=merge_bilateral,merge_dorsoventral=False,
                merge_numbered=True,merge_AWC=True,
                ds_exclude_tags="mutant", #ds_tags="D20", 
                signal="green",signal_kwargs=signal_kwargs)
                
funa_unc31 = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=merge_bilateral,merge_dorsoventral=False,
                merge_numbered=True,merge_AWC=True,
                ds_tags="unc31",
                signal="green",signal_kwargs=signal_kwargs)
                
_, occ2_wt = funa_wt.get_occurrence_matrix(inclall=inclall)
_, occ2_unc31 = funa_unc31.get_occurrence_matrix(inclall=inclall)

ai_RID = funa_wt.ids_to_i(jid)
ai_i = funa_wt.ids_to_i(iid)

t = np.linspace(-10,30,120)
t0 = np.argmin(np.abs(t-0))

resp_wt = []
resp_unc31 = []

peak_t_wt = []
peak_t_unc31 = []

max_wt = []
max_unc31 = []

min_wt = []
min_unc31 = []

avg_wt = []
avg_unc31 = []


for io in np.arange(len(occ2_wt[ai_i,ai_RID])):
    o = occ2_wt[ai_i,ai_RID][io]
    ds = o["ds"]
    ie = o["stim"]
    i = o["resp_neu_i"]
    
    time,time2,i0,i1 = funa_wt.fconn[ds].get_time_axis(ie,True)
    shift_vol = funa_wt.fconn[ds].shift_vols[ie]
    
    y = funa_wt.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,i]
    nan_mask = funa_wt.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
    
    pre = np.average(y[:shift_vol])
    y = (y-pre)/pre
    
    if np.max(y)>2:continue
                                
    if pp.Fconn.nan_ok(nan_mask,pp.Funatlas.nan_th*len(nan_mask)):
        max_wt.append(np.nanmax(y))
        min_wt.append(np.nanmin(y))
        avg_wt.append(np.nanmean(y))
        peak_t_wt.append(np.nanargmax((y)))
        
        resp_wt.append(np.interp(t,time,y))

resp_wt = np.array(resp_wt)
    
if sort_max:
    resp_wt = resp_wt[np.argsort(max_wt)[::-1]]
elif sort_t:
    resp_wt = resp_wt[np.argsort(peak_t_wt),:]
elif sort_min:
    resp_wt = resp_wt[np.argsort(min_wt),:]
elif sort_avg:
    resp_wt = resp_wt[np.argsort(avg_wt)[::-1],:]
    
for io in np.arange(len(occ2_unc31[ai_i,ai_RID])):
    o = occ2_unc31[ai_i,ai_RID][io]
    ds = o["ds"]
    ie = o["stim"]
    i = o["resp_neu_i"]
    
    time,time2,i0,i1 = funa_unc31.fconn[ds].get_time_axis(ie,True)
    shift_vol = funa_unc31.fconn[ds].shift_vols[ie]
    
    y = funa_unc31.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,i]
    nan_mask = funa_unc31.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
    
    pre = np.average(y[:shift_vol])
    y = (y-pre)/pre
    
    if pp.Fconn.nan_ok(nan_mask,pp.Funatlas.nan_th*len(nan_mask)):
        max_unc31.append(np.nanmax(y))
        min_unc31.append(np.nanmin(y))
        avg_unc31.append(np.nanmean(y))
        peak_t_unc31.append(np.nanargmax((y)))
            
        resp_unc31.append(np.interp(t,time,y))
        
resp_unc31 = np.array(resp_unc31)
    
if sort_max:
    resp_unc31 = resp_unc31[np.argsort(max_unc31)[::-1]]
elif sort_t:
    resp_unc31 = resp_unc31[np.argsort(peak_t_unc31)]
elif sort_min:
    resp_unc31 = resp_unc31[np.argsort(min_unc31)]
elif sort_avg:
    resp_unc31 = resp_unc31[np.argsort(avg_unc31)[::-1]]

if not relative:
    print("Bypassing vmax")
    max1 = np.sort(np.ravel(np.abs(resp_wt)))[-20]
    max2 = np.sort(np.ravel(np.abs(resp_unc31)))[-20]
    print(max1,max2)
    vmax = np.around(max(max1,max2),1)
    #vmax = max(np.max(np.abs(resp_wt)),np.max(np.abs(resp_unc31)))
    #vmax = 0.25
    
fig = plt.figure(1,constrained_layout=True)
gs = fig.add_gridspec(3,2)
#ax1 = fig.add_subplot(121)
#ax2 = fig.add_subplot(122)
ax1 = fig.add_subplot(gs[1:,0])
ax2 = fig.add_subplot(gs[1:,1])
ax1t = fig.add_subplot(gs[0,0])
ax2t = fig.add_subplot(gs[0,1],sharey=ax1t)
im = ax1.imshow(resp_wt,aspect="auto",interpolation="nearest",vmin=-vmax,vmax=vmax,cmap="coolwarm")
ax2.imshow(resp_unc31,aspect="auto",interpolation="nearest",vmin=-vmax,vmax=vmax,cmap="coolwarm")
ax1.axvline(t0,color="#000000")
ax2.axvline(t0,color="#000000")
ax1t.axvline(t0,color="#000000")
ax1t.axhline(0,color="#000000")
ax2t.axvline(t0,color="#000000")
ax2t.axhline(0,color="#000000")

nwt = len(resp_wt)
nunc31 = len(resp_unc31)
resp_wt_avg = np.average(resp_wt,axis=0)
try:
    resp_wt_avg_1 = np.average(resp_wt[:nwt//2],axis=0)
    resp_wt_avg_2 = np.average(resp_wt[nwt//2:],axis=0)
except:
    resp_wt_avg_1,resp_wt_avg_2 = np.ones(len(resp_wt[0]))*np.nan
resp_wt_max = np.max(resp_wt,axis=0)
resp_wt_min = np.min(resp_wt,axis=0)
resp_wt_std = np.std(resp_wt,axis=0)
resp_unc31_avg = np.average(resp_unc31,axis=0)
try:
    resp_unc31_avg_1 = np.average(resp_unc31[:nunc31//2],axis=0)
    resp_unc31_avg_2 = np.average(resp_unc31[nunc31//2:],axis=0)
except:
    resp_unc31_avg_1,resp_unc31_avg_2 = np.ones(len(resp_unc31[0]))*np.nan
resp_unc31_max = np.max(resp_unc31,axis=0)
resp_unc31_min = np.min(resp_unc31,axis=0)
resp_unc31_std = np.std(resp_unc31,axis=0)
ax1t.plot(resp_wt_avg)
ax2t.plot(resp_unc31_avg)
#ax1t.fill_between(np.arange(len(resp_wt_avg)),resp_wt_min,resp_wt_max,color="k",alpha=0.1)
#ax2t.fill_between(np.arange(len(resp_unc31_avg)),resp_unc31_min,resp_unc31_max,color="k",alpha=0.1)
ax1t.fill_between(np.arange(len(resp_wt_avg)),resp_wt_avg-resp_wt_std,resp_wt_avg+resp_wt_std,color="k",alpha=0.1)
ax2t.fill_between(np.arange(len(resp_unc31_avg)),resp_unc31_avg-resp_unc31_std,resp_unc31_avg+resp_unc31_std,color="k",alpha=0.1)
#ax1t.plot(resp_wt_avg_1)
#ax1t.plot(resp_wt_avg_2)
#ax2t.plot(resp_unc31_avg_1)
#ax2t.plot(resp_unc31_avg_2)

min1 = np.min(resp_wt_avg-resp_wt_std)
min2 = np.min(resp_unc31_avg-resp_unc31_std)
ymin = min(min1,min2)

max1 = np.max(resp_wt_avg+resp_wt_std)
max2 = np.max(resp_unc31_avg+resp_unc31_std)
ymax = max(max1,max2)

ax1t.set_xticks([])
ax2t.set_xticks([])
ax1t.set_ylim(ymin,ymax)
ax1t.set_ylabel("$\Delta F/F$")

#xticks = [0,20,40,60,80,100]
xticks = [00,40,80]
xticks = np.zeros(3,dtype=int)
xticks[0] = np.argmin(np.abs(t+10))
xticks[1] = np.argmin(np.abs(t))
xticks[2] = np.argmin(np.abs(t-20))
print("\n\n\n\n\n\n")
print(xticks)
print("\n\n\n\n\n\n")
ax1.set_xticks(xticks)
ax1.set_xticklabels((np.around(t[xticks],0)).astype(int))
de = int(len(resp_wt)//4)
if de==0: de=1
ax1.set_yticks(np.arange(len(resp_wt))[::de])
ax1.set_yticklabels([str(ti+1) for ti in np.arange(len(resp_wt))[::de]])
ax2.set_xticks(xticks)
ax2.set_xticklabels((np.around(t[xticks],0)).astype(int))
de = int(len(resp_unc31)//4)
if de==0: de=1
ax2.set_yticks(np.arange(len(resp_unc31))[::de])
ax2.set_yticklabels([str(ti+1) for ti in np.arange(len(resp_unc31))[::de]])

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Trials")
ax2.set_xlabel("Time (s)")

ax1.set_title(jid+" -> "+iid+" (WT)",fontsize=14)
ax2.set_title(jid+" -> "+iid+" ($\mathit{unc-31})$",fontsize=14)

cax = fig.colorbar(im,ax=[ax1,ax2],location="bottom")
lbl = "$\Delta F/F$" if relative else "$\Delta F$"
cax.ax.set_xlabel(lbl)
cax.ax.set_xticks([-vmax,0,vmax])

stamp = re.sub("(.{40})", "\\1\n", " ".join(sys.argv), 0, re.DOTALL)
#pp.provstamp(ax1,-.1,-.1,stamp)
#fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/"+jid+"->"+iid+".svg",bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/"+jid+"->"+iid+".png",bbox_inches="tight",dpi=300)
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/"+jid+"->"+iid+".png",bbox_inches="tight",dpi=300)
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/"+jid+"->"+iid+".pdf",bbox_inches="tight",dpi=300)
