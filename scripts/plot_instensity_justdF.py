import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit

sort = "--sort" in sys.argv
pop_nans = "--pop-nans" in sys.argv
stamp = "--no-stamp" not in sys.argv

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "mutant"
ds_exclude_i = []
inclall_occ = "--inclall-occ" in sys.argv
signal_range = None
smooth_mode = "sg_causal"
smooth_n = 13
smooth_poly = 1
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
nan_th = 0.3
save = "--no-save" not in sys.argv
save_cache = "--save-cache" in sys.argv
vmax = 0.4
cmap = "Spectral_r"
two_min_occ = "--two-min-occ" in sys.argv
enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv
to_paper = "--to-paper" in sys.argv
plot = "--no-plot" not in sys.argv
figsize = (12,10)

for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--nan-th": nan_th = float(sa[1])
    if sa[0] == "--ds-exclude-tags": 
        ds_exclude_tags=sa[1]
        if ds_exclude_tags == "None": ds_exclude_tags=None
    if sa[0] == "--ds-tags": ds_tags=sa[1]
    if sa[0] == "--ds-exclude-i": ds_exclude_i = [int(sb) for sb in sa[1].split(",")]
    if sa[0] == "--signal-range":
        sb = sa[1].split(",")
        signal_range = [int(sbi) for sbi in sb]
    if sa[0] == "--vmax": vmax = float(sa[1])
    if sa[0] == "--cmap": cmap = sa[1]
    if sa[0] == "--figsize": figsize = [int(sb) for sb in sa[1].split(",")]
    
    
# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,          
                 "photobl_appl":True,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file}

funa = pp.Funatlas.from_datasets(
                ds_list,
                merge_bilateral=merge_bilateral,
                merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                signal_kwargs=signal_kwargs,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck,
                verbose=False)


occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ,req_auto_response=req_auto_response)
# If occ2 needs to be filtered
occ1,occ2 = funa.filter_occ12_from_sysargv(occ2,sys.argv)
                 
occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=req_auto_response)

mappa = np.zeros((funa.n_neurons,funa.n_neurons))*np.nan
count = np.zeros((funa.n_neurons,funa.n_neurons))

for ai in np.arange(funa.n_neurons):
    for aj in np.arange(funa.n_neurons):
        if np.isnan(mappa[ai,aj]) and occ3[ai,aj]>0:
            mappa[ai,aj] = 0.0
        
        if two_min_occ:
            if occ1[ai,aj]<2: continue
        
        ys = []
        times = []
        confidences = []
        for occ in occ2[ai,aj]:
            ds = occ["ds"]
            if ds in ds_exclude_i: continue
            ie = occ["stim"]
            i = occ["resp_neu_i"]
            
            # Build the time axis
            i0 = funa.fconn[ds].i0s[ie]
            i1 = funa.fconn[ds].i1s[ie]
            shift_vol = funa.fconn[ds].shift_vol
            
            y = funa.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,i]
            nan_mask = funa.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
                                         
            if np.sum(nan_mask)>nan_th*len(y): continue
                                         
            if signal_range is None:    
                pre = np.average(y[:shift_vol])                                 
                if pre == 0: continue
                dy = np.average( y[shift_vol:] - pre )/pre
            else:
                #std = np.std(y[:shift_vol-signal_range[0]])
                pre = np.average(y[:shift_vol])                                 
                #dy = np.average(y[shift_vol-signal_range[0]:shift_vol+signal_range[1]+1] - pre)
                dy = np.average(np.abs(y[shift_vol-signal_range[0]:shift_vol+signal_range[1]+1] - pre))
                dy /= pre
            
            if np.isnan(mappa[ai,aj]):
                mappa[ai,aj] = dy
            else:
                mappa[ai,aj] += dy
            count[ai,aj] += 1
            
mappa[count>0] /= count[count>0]

mappa_full = np.copy(mappa)
mappa = funa.reduce_to_head(mappa)

if sort:
    mappa,sorter,lim = funa.sort_matrix_nans(mappa,axis=-1,return_all=True)
    sorter_i = sorter_j = sorter
    
elif pop_nans:
    mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(mappa,return_all=True)
        
else:
    sorter_i = sorter_j = np.arange(mappa.shape[-1])
    lim = None
    

fig2 = plt.figure(1, figsize=figsize)
ax2 = fig2.add_subplot()
ax2.imshow(0. * np.ones_like(mappa), cmap="Greys", vmax=1, vmin=0)
#blank_mappa = np.copy(fraction_respond_head)
#blank_mappa[~np.isnan(fraction_respond_head)] = 0.2
#ax2.imshow(blank_mappa, cmap="Greys", vmin=0, vmax=1)
im = ax2.imshow(mappa, cmap=cmap, vmin=-0.4, vmax=0.4, interpolation="nearest")
diagonal = np.diag(np.diag(np.ones_like(mappa)))
new_diagonal = np.zeros_like(mappa)
new_diagonal[np.where(diagonal == 1)] = 1
ax2.imshow(new_diagonal, cmap="binary", vmin=0, vmax=1, alpha=new_diagonal, interpolation="nearest")
plt.colorbar(im, label = r'$\langle\Delta F/F\rangle_t$')
ax2.set_xlabel("stimulated",fontsize=30)
ax2.set_ylabel("responding",fontsize=30)
ax2.set_xticks(np.arange(len(sorter_j)))
ax2.set_yticks(np.arange(len(sorter_i)))
ax2.set_xticklabels(funa.head_ids[sorter_j], fontsize=4, rotation=90)
ax2.set_yticklabels(funa.head_ids[sorter_i], fontsize=4)
ax2.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax2.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
ax2.set_xlim(-0.5, lim)
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/intensitymap_justDF.pdf")
plt.savefig("/home/sdvali/Desktop/Maps/intensitymap_justDF.pdf")
print("done")