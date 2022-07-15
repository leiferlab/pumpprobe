import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp


ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": True}
                 

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=True,
                                 ds_tags=None,ds_exclude_tags="mutant",
                                 verbose=False)

RID = funa.ids_to_i(["RID"])[0]
                                 
_,occ2 = funa.get_occurrence_matrix(req_auto_response=False,inclall=True)
occ3 = funa.get_observation_matrix()

p,_,_ = funa.get_kolmogorov_smirnov_p(occ2)

maxabsdff = np.zeros(funa.n_neurons)
for ai in np.arange(funa.n_neurons):
    maxabsdff_ = []
    for io in np.arange(len(occ2[ai,RID])):
        o = occ2[ai,RID][io]
        ds = o["ds"]
        ie = o["stim"]
        i = o["resp_neu_i"]
        
        time,time2,i0,i1 = funa.fconn[ds].get_time_axis(ie,True)
        shift_vol = funa.fconn[ds].shift_vols[ie]
        
        seg = funa.sig[ds].get_segment(
                                    i0,i1,shift_vol,
                                    baseline=False,normalize="")[:,i]
        nan = funa.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
                                    
        baseline = np.nanmean(seg[shift_vol//2:shift_vol])
        y = (seg-baseline)
        y = y/baseline
        
        maxabsdff_.append(np.nanmax(np.abs(y)))
    maxabsdff[ai] = np.nanmean(maxabsdff_)
    
argsort = np.argsort(maxabsdff)[::-1]
argsort2 = np.argsort(p[:,RID])
psorted = np.sort(p[:,RID])

k = 0
while psorted[k]<0.2:
    print(funa.neuron_ids[argsort2[k]],psorted[k])
    k+=1

fig = plt.figure(1)
ax = fig.add_subplot(111)
axt = ax.twinx()
ax.plot(maxabsdff[argsort])
axt.plot(occ3[:,RID][argsort],alpha=0.4)
ax.set_xticks(np.arange(funa.n_neurons))
ax.set_xticklabels(funa.neuron_ids[argsort])
fig.tight_layout()

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.plot(psorted)
ax.set_xticks(np.arange(funa.n_neurons))
ax.set_xticklabels(funa.neuron_ids[argsort2])
fig.tight_layout()
plt.show()
