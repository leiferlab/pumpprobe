import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

sort = "--sort" in sys.argv
pop_nans = "--pop-nans" in sys.argv
stamp = "--no-stamp" not in sys.argv
SIM = "--sim" in sys.argv

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "mutant"
ds_exclude_i = []
inclall_occ = "--inclall-occ" in sys.argv
signal_range = None
smooth_mode = "sg_causal"
smooth_n = 13
smooth_poly = 1
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
nan_th = pp.Funatlas.nan_th
save = "--no-save" not in sys.argv
vmax = 0.4
cmap = 'Oranges_r'

enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv
to_paper = "--to-paper" in sys.argv
plot = "--no-plot" not in sys.argv
figsize = (12,10)
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--nan-th": nan_th = float(sa[1])
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
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
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only,
                 "photobl_appl":True}

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


if inclall_occ: inclall_occ2 = occ2
qvalues, tost_q = funa.get_kolmogorov_smirnov_q(inclall_occ2,
                                                return_tost=True,tost_th=1.2)

qvalues_head = funa.reduce_to_head(qvalues)
tost_q_head = funa.reduce_to_head(tost_q)
if sort:
    qvalues_head,sorter,lim = funa.sort_matrix_nans(qvalues_head,axis=-1,return_all=True)
    sorter_i = sorter_j = sorter
elif pop_nans:
    qvalues_head,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(qvalues_head,return_all=True)
else:
    sorter_i = sorter_j = np.arange(qvalues_head.shape[-1])
    lim = None

tost_q_head = tost_q_head[sorter_i][:,sorter_j]

fig3 = plt.figure(1, figsize=figsize)
ax3 = fig3.add_subplot()
ax3.imshow(0. * np.ones_like(qvalues_head), cmap="Greys", vmax=1, vmin=0)
#blank_mappa = np.copy(occ3_head)
#blank_mappa[~np.isnan(occ3_head)] = 0.2
#ax3.imshow(blank_mappa, cmap="Greys", vmin=0, vmax=1)
im = ax3.imshow(qvalues_head, cmap=cmap, vmin=0, vmax=np.nanmax(qvalues_head), interpolation="nearest")
diagonal = np.diag(np.diag(np.ones_like(qvalues_head)))
new_diagonal = np.zeros_like(qvalues_head)
new_diagonal[np.where(diagonal == 1)] = 1
ax3.imshow(new_diagonal, cmap="binary", vmin=0, vmax=1, alpha=new_diagonal, interpolation="nearest")
plt.colorbar(im, label = r'q value')
ax3.set_xlabel("stimulated",fontsize=30)
ax3.set_ylabel("responding",fontsize=30)
ax3.set_xticks(np.arange(len(sorter_j)))
ax3.set_yticks(np.arange(len(sorter_i)))

ax3.set_xticklabels(funa.head_ids[sorter_j], fontsize=4, rotation=90)
ax3.set_yticklabels(funa.head_ids[sorter_i], fontsize=4)
ax3.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax3.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
ax3.set_xlim(-0.5, lim+0.5)
if merge_bilateral:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/qvalues_merged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/qvalues_merged.pdf",dpi=300, bbox_inches="tight")
else:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/qvalues_unmerged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/qvalues_unmerged.pdf",dpi=300, bbox_inches="tight")
    f = open("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/qvalues_unmerged.txt","w")
    ll_ = funa.head_ids
    f.write(","+",".join(ll_[sorter_j])+"\n")
    s = ""
    for i in np.arange(qvalues_head.shape[0]):
        s += ll_[sorter_i][i]+","+",".join(qvalues_head[i].astype(str)+"\n")
    s = s[:-1]
    f.write(s)
    
fig3.clf()

tqhmax = np.nanmax(tost_q_head)
div = int(0.05/tqhmax*256)

colors1 = plt.cm.autumn(np.linspace(0., 1, div))
colors2 = plt.cm.viridis_r(np.linspace(0, 1, 256-div))
colors = np.vstack((colors1, colors2))
mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
fig = plt.figure(1, figsize=figsize)
ax = fig.add_subplot()
ax.imshow(0. * np.ones_like(tost_q_head), cmap="Greys", vmax=1, vmin=0)
im = ax.imshow(tost_q_head, cmap=mymap, vmin=0, vmax=tqhmax, interpolation="nearest")
diagonal = np.diag(np.diag(np.ones_like(tost_q_head)))
new_diagonal = np.zeros_like(tost_q_head)
new_diagonal[np.where(diagonal == 1)] = 1
ax.imshow(new_diagonal, cmap="binary", vmin=0, vmax=1, alpha=new_diagonal, interpolation="nearest")
plt.colorbar(im, label = r'q_{eq} value')
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(sorter_j)))
ax.set_yticks(np.arange(len(sorter_i)))

ax.set_xticklabels(funa.head_ids[sorter_j], fontsize=4, rotation=90)
ax.set_yticklabels(funa.head_ids[sorter_i], fontsize=4)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
ax.set_xlim(-0.5, lim+0.5)
if merge_bilateral:
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/tost_q_merged.pdf",dpi=300, bbox_inches="tight")
else:
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/tost_q_unmerged.pdf",dpi=300, bbox_inches="tight")
