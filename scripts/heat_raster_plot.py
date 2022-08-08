import numpy as np, matplotlib.pyplot as plt, sys, re
import pumpprobe as pp, wormdatamodel as wormdm

#Typical command: heat_raster_plot.py --j:ASH --i:AVA --inclall-occ --sort-avg --relative

headless = "--headless" in sys.argv
inclall = "--inclall-occ" in sys.argv
sort_max = "--sort-max" in sys.argv
sort_avg = "--sort-avg" in sys.argv
relative = "--relative" in sys.argv
nomerge  = "--nomerge" in sys.argv
paired = "--paired" in sys.argv
unc31 = "--unc31" in sys.argv
req_auto_response = "--req_auto_response" in sys.argv
iid = "" #Downstream neuron
jid = "" #Stimulated neuron
vmax = 0.5
matchless_nan_th = 1.
for s in sys.argv:
    sa = s.split(":")
    if sa[0] in ["-i","--i"] : iid=sa[1] #Downstream neuron
    if sa[0] in ["-j","--j"] : jid=sa[1] #Stimulated neuron
    if sa[0] == "--vmax": vmax=float(sa[1])
    if sa[0] == "--matchless-nan-th": matchless_nan_th=float(sa[1])

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "matchless_nan_th": matchless_nan_th}
                 #"matchless_nan_th_from_file": True}


merge_bilateral = not(nomerge)
merge_AWC = not(nomerge)

if unc31:
    funa_wt = pp.Funatlas.from_datasets(
                    ds_list,merge_bilateral=merge_bilateral,merge_dorsoventral=False,
                    merge_numbered=False,merge_AWC=merge_AWC,
                    ds_tags="unc31",
                    signal_kwargs=signal_kwargs,verbose=False, enforce_stim_crosscheck=False)
    unc31_flag="unc31"
else:
    funa_wt = pp.Funatlas.from_datasets(
                    ds_list,merge_bilateral=merge_bilateral,merge_dorsoventral=False,
                    merge_numbered=False,merge_AWC=merge_AWC,
                    ds_exclude_tags="mutant",
                    signal_kwargs=signal_kwargs,verbose=False, enforce_stim_crosscheck=False)
    unc31_flag=""
                

                
_, occ2_wt = funa_wt.get_occurrence_matrix(inclall=inclall, req_auto_response=req_auto_response)

ai_j = funa_wt.ids_to_i(jid)
ai_i = funa_wt.ids_to_i(iid)

duration_before=-10
duration_after=31
t = np.linspace(duration_before,duration_after,2*(duration_after-duration_before))
t0 = np.argmin(np.abs(t-0))

resp_wt = []
resp_upstream = []
peak_t_wt = []
max_wt = []
min_wt = []
avg_wt = []

instances = len(occ2_wt[ai_i, ai_j])
assert instances > 0, "No observations of neuron pair found"

for io in np.arange(instances):
    o = occ2_wt[ai_i, ai_j][io]
    ds = o["ds"] #dataset (e.g. the recording number)
    ie = o["stim"] #stimulation instance
    i = o["resp_neu_i"] #responding neuron
    j = funa_wt.fconn[ds].stim_neurons[ie] #stimulated neuron
    
    time,time2,i0,i1 = funa_wt.fconn[ds].get_time_axis(ie,True)
    shift_vol = funa_wt.fconn[ds].shift_vols[ie]
    
    seg = funa_wt.sig[ds].get_segment(
                                i0,i1,shift_vol,
                                baseline=False,normalize="")[:,i]
    seg_upstream = funa_wt.sig[ds].get_segment(
                                i0,i1,shift_vol,
                                baseline=False,normalize="")[:,j]
    nan = funa_wt.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
    nan_upstream = funa_wt.sig[ds].get_segment_nan_mask(i0,i1)[:,j]

                                
    baseline = np.nanmean(seg[shift_vol//2:shift_vol])
    baseline_upstream = np.nanmean(seg_upstream[shift_vol//2:shift_vol])

    y = (seg-baseline)
    y_upstream = (seg_upstream - baseline_upstream)
    if relative:
        y = y/baseline
        y_upstream = y_upstream/baseline_upstream

    
    if np.sum(nan)<.3*len(nan):
        
        max_wt.append(np.nanmax(y[shift_vol:]))
        min_wt.append(np.nanmin(y[shift_vol:]))
        avg_wt.append(np.nanmean(y[shift_vol:shift_vol+duration_after]))
        peak_t_wt.append(np.nanargmax((y[shift_vol:])))
        
        resp_wt.append(np.interp(t,time,y))
        resp_upstream.append(np.interp(t,time,y_upstream))
        
resp_wt = np.array(resp_wt)
resp_upstream = np.array(resp_upstream)

#calculcate the average response
av_resp = np.average(resp_wt,0)
av_resp_upstream = np.average(resp_upstream,0)
#estimate excitatory or inhibitory
if np.sum(av_resp[t>0]) >0:
    sortorder = -1 #excitatory
else:
    sortorder = 1 #inhibitory

if sort_max:
    resp_wt = resp_wt[np.argsort(max_wt)[::sortorder]]
    resp_upstream = resp_upstream[np.argsort(max_wt)[::sortorder]] #sort upstream to match response
elif sort_avg:
    resp_wt = resp_wt[np.argsort(avg_wt)[::sortorder],:]
    resp_upstream = resp_upstream[np.argsort(avg_wt)[::sortorder],:] #sort upstream to match response

if paired:
    cols = 2
else:
    cols = 1


#Let's think about spacing of the figure. Currently we  have 3.5 units per colun,
# and (3/4)*5. vertical units when we have 14 rows + (1/4)*5.5 or 1.375 vertical units for the average trace
# We can probably go even tighter on the rows.. to say 4. So that works out to 0.285 units per row
fig_width = 3.5*cols
hieght_per_row = 0.28
trace_height = 1.37 #in abs units
buffer_rows = 2
fig_height = trace_height + len(resp_wt)*hieght_per_row + buffer_rows*hieght_per_row
trace_height_percent = trace_height / fig_height
spatial_resolution = 100
trace_height_gs_units = int(np.round(trace_height_percent*spatial_resolution))

#fig = plt.figure(1,constrained_layout=True,figsize=(3.5*cols,5.5))
fig = plt.figure(1, constrained_layout=True, figsize=(fig_width, fig_height))
#gs =fig.add_gridspec(4,cols)
gs =fig.add_gridspec(spatial_resolution,cols)


#Plot the average trace
ax2 = fig.add_subplot(gs[0:trace_height_gs_units,cols-1])
error = np.std(resp_wt,0)
ax2.plot(t,av_resp,linewidth=2)
ax2.axhline(0,color='black')
ax2.fill_between(t, av_resp-error, av_resp+error,alpha=.1)
lbl = "$\Delta F/F_0$" if relative else "$\Delta F$"
ax2.set_ylabel(lbl)
ax2.set_xlim(duration_before-.25,duration_after-.25)
ax2.axvline(0,color='black')


if paired: #plot the upstream average trace for only those instances when the downstream was also observed
    ax2us = fig.add_subplot(gs[0:trace_height_gs_units, cols - 2], sharey=ax2)
    error_upstream = np.std(resp_upstream, 0)
    ax2us.plot(t, av_resp_upstream, linewidth=2)
    ax2us.axhline(0, color='black')
    ax2us.fill_between(t, av_resp_upstream - error_upstream, av_resp_upstream + error_upstream, alpha=.1)
    ax2us.set_ylabel(lbl)
    ax2us.set_xlim(duration_before - .25, duration_after - .25)
    ax2us.axvline(0, color='black')
    ax2.set_title(iid + " (Resp)")
    ax2us.set_title(jid + " (Stim)")
    #remove downstream label
    ax2.set_ylabel("")
else:
    fig.suptitle(jid+" -> "+iid+" (WT)")



#Plot the heat raster
ax1 = fig.add_subplot(gs[trace_height_gs_units+1:,cols-1])
im = ax1.imshow(resp_wt,aspect="auto",interpolation="nearest",vmin=-vmax,vmax=vmax,cmap="coolwarm")
ax1.axvline(t0,color='black')
if paired:
    ax1us = fig.add_subplot(gs[trace_height_gs_units+1:, cols - 2])
    imus = ax1us.imshow(resp_upstream, aspect="auto", interpolation="nearest", vmin=-vmax, vmax=vmax, cmap="coolwarm")
    ax1us.axvline(t0, color='black')


xticks = [0,20,40,60,80]
ax1.set_xticks(xticks)
ax2.set_xticks(np.array(xticks)/2-10)
ax1.set_xticklabels((np.around(t[xticks],0)).astype(int))
ax1.set_yticks(np.arange(len(resp_wt)))
ax1.set_yticklabels(np.arange(len(resp_wt))+1)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Sorted\n Trials")


if paired:
    ax1us.set_xticks(xticks)
    ax2us.set_xticks(np.array(xticks) / 2 - 10)
    ax1us.set_xticklabels((np.around(t[xticks], 0)).astype(int))
    ax1us.set_yticks(np.arange(len(resp_upstream)))
    ax1us.set_yticklabels(np.arange(len(resp_upstream)) + 1)
    ax1us.set_xlabel("Time (s)")
    ax1us.set_ylabel("Sorted\n Trials")
    #Get rid of ticks and labels downstream
    ax1.set_ylabel("")
    ax1.set_yticklabels([])





cax = fig.colorbar(im,ax=[ax1,ax2],location="right", shrink=0.5, ticks=[-vmax, 0, vmax])
cax.ax.set_yticklabels(['< -'+str(vmax), '0', '> '+str(vmax)])
cax.ax.set_xlabel(lbl)

import shlex
stamp = " ".join(map(shlex.quote, sys.argv[0:]))

from pathlib import Path
script_dir=Path(__file__).parent #Get path to this script
fig.savefig( (script_dir /  (jid+'_'+iid+'_'+unc31_flag+'_heat_raster.pdf')),bbox_inches="tight",metadata=pp.provenance.pdf_metadata(stamp))
fig.savefig(  (script_dir /  (jid+'_'+iid+'_'+unc31_flag+'_heat_raster.png')),metadata=pp.provenance.png_metadata(stamp), bbox_inches="tight",dpi=300)
if not headless: plt.show()

