import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit

sort = "--sort" in sys.argv
pop_nans = "--pop-nans" in sys.argv
stamp = "--no-stamp" not in sys.argv
SIM = "--sim" in sys.argv

use_kernels = "--use-kernels" in sys.argv
drop_saturation_branches = "--drop-saturation-branches" in sys.argv

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
nan_th = 0.3
save = "--no-save" not in sys.argv
save_cache = "--save-cache" in sys.argv
use_rise_times = "--use-decay-times" not in sys.argv
vmax_rise = 2.0 
vmin_rise = 0 
show_diag = "--show-diag" in sys.argv
cmap = "coolwarm"
two_min_occ = "--two-min-occ" in sys.argv
alpha_qvalues = "--alpha-qvalues" in sys.argv
alpha_kolmogorov_smirnov = "--alpha-kolmogorov-smirnov" in sys.argv
alpha_occ1 = not alpha_qvalues and not alpha_kolmogorov_smirnov
alphamax_set = None
alphamin_set = None
alphalbl_set = None
alphaticklabels = None
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
    if sa[0] == "--vmax-rise": vmax_rise = float(sa[1])
    if sa[0] == "--cmap": cmap = sa[1]
    if sa[0] == "--alpha-max": alphamax_set = float(sa[1])
    if sa[0] == "--alpha-min": alphamin_set = float(sa[1])
    if sa[0] == "--alpha-lbl": alphalbl_set = sa[1]
    if sa[0] == "--alpha-tick-labels": alphaticklabels = [sb for sb in sa[1].split(",")]
    if sa[0] == "--figsize": figsize = [int(sb) for sb in sa[1].split(",")]

# To determine which control distribution to use
strain = "unc31" if ds_tags == "unc31" else ""
    
# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,          
                 "photobl_appl":True,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

funa = pp.Funatlas.from_datasets(
                ds_list,
                merge_bilateral=merge_bilateral,
                merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                signal_kwargs=signal_kwargs,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck,
                verbose=False)

print(len(funa.ds_list))

#funa.export_to_txt("/projects/LEIFER/francesco/funatlas/exported_data/")

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ,req_auto_response=req_auto_response)
# If occ2 needs to be filtered
occ1,occ2 = funa.filter_occ12_from_sysargv(occ2,sys.argv)
                 
occ3 = funa.get_observation_matrix(req_auto_response=req_auto_response)

# Setting the transparency of the imshows
if alpha_occ1:
    occ1_head = funa.reduce_to_head(occ1)
    occ1_head = occ1_head/np.max(occ1_head)
    alphalbl = "n responses\n(norm to max)"
    alphamin,alphamax = 0.,0.1#0.25, 1
    if alphamax_set is not None: alphamax = alphamax_set
    if alphamin_set is not None: alphamin = alphamin_set
    alphas = np.clip((occ1_head-alphamin)/(alphamax-alphamin),0,1)
elif alpha_qvalues:
    #qvalues = funa.get_qvalues(occ1,occ3,False)
    qvalues = funa.get_qvalues2(merge_bilateral,req_auto_response)
    print("USING get_qvalues2")
    if SIM:
        qvalues_head = funa.reduce_to_SIM_head(qvalues)
    else:
        qvalues_head = funa.reduce_to_head(qvalues)
    #print(np.sum( (qvalues_head<0.2))/np.prod(qvalues_head.shape))
    qvalues_head[np.isnan(qvalues_head)] = 1.
    alphas = (1.-qvalues_head)
    alphalbl = "1-q"
    alphamin,alphamax = 0.,0.9#0.75
    if alphamax_set is not None: alphamax = alphamax_set
    if alphamin_set is not None: alphamin = alphamin_set
    alphas = np.clip((alphas-alphamin)/(alphamax-alphamin),0,1)
elif alpha_kolmogorov_smirnov:
    if inclall_occ: inclall_occ2 = occ2
    else: _, inclall_occ2 = funa.get_occurrence_matrix(inclall=True,req_auto_response=req_auto_response)
    qvalues = funa.get_kolmogorov_smirnov_q(inclall_occ2,strain=strain)
    qvalues_orig = np.copy(qvalues)
    print("Got qvalues")
    qvalues[np.isnan(qvalues)] = 1.
    if SIM:
        qvalues_head = funa.reduce_to_SIM_head(qvalues)
    else:
        qvalues_head = funa.reduce_to_head(qvalues)
    alphas = (1.-qvalues_head)
    alphalbl = "1-q"
    if alphalbl_set is not None: alphalbl = alphalbl_set
    alphamin,alphamax = 0.,1.0
    if alphamax_set is not None: alphamax = alphamax_set
    if alphamin_set is not None: alphamin = alphamin_set
    alphas = np.clip((alphas-alphamin)/(alphamax-alphamin),0,1)

# Get timescales
if not use_kernels:
    time1 = np.linspace(0,30,1000)
    time2 = np.linspace(0,200,1000)
else:
    time1 = np.linspace(0,30,1000)
    time2 = np.linspace(0,10,1000)
dFF = funa.get_max_deltaFoverF(occ2,time1,nans_to_zero=True)
if use_rise_times:
    rise_times = funa.get_eff_rise_times(occ2,time2,use_kernels,drop_saturation_branches)
    # Taking the weighted average with weights dFF*conf
    avg_rise_times = funa.weighted_avg_occ2style2(rise_times,[dFF])
    mappa = avg_rise_times
else:
    decay_times = funa.get_eff_decay_times(occ2,time2,use_kernels,drop_saturation_branches)
    # Taking the weighted average with weights dFF*conf
    avg_decay_times = funa.weighted_avg_occ2style2(decay_times,[dFF])
    mappa = avg_decay_times

mappa_full = np.copy(mappa)
if SIM:
    mappa = funa.reduce_to_SIM_head(mappa)
else:
    mappa = funa.reduce_to_head(mappa)
if sort:
    mappa,sorter,lim = funa.sort_matrix_nans(mappa,axis=-1,return_all=True)
    sorter_i = sorter_j = sorter
    alphas = alphas[sorter][:,sorter]
elif pop_nans:
    if SIM:
        mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans_SIM(mappa)
        alphas = alphas[sorter_i][:,sorter_j]
    else:
        mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(mappa,return_all=True)
        alphas = alphas[sorter_i][:,sorter_j]
else:
    sorter_i = sorter_j = np.arange(mappa.shape[-1])
    lim = None

if plot:    
    fig1 = plt.figure(1,figsize=figsize)
    gs = fig1.add_gridspec(1,10)
    ax = fig1.add_subplot(gs[0,:9])
    cax = fig1.add_subplot(gs[0,9:])
    #ax.fill_between(np.arange(mappa.shape[1]), 0, y2=mappa.shape[0], hatch='//////', zorder=0, fc='white')
    ax.imshow(0.*np.ones_like(mappa),cmap="Greys",vmax=1,vmin=0)
    blank_mappa = np.copy(mappa)
    blank_mappa[~np.isnan(mappa)] = 0.1
    ax.imshow(blank_mappa,cmap="Greys",vmin=0,vmax=1)
    im = ax.imshow(mappa,cmap=cmap,vmin=vmin_rise,vmax=vmax_rise,alpha=alphas,interpolation="nearest")
    if not show_diag:
        diagonal = np.diag(np.diag(np.ones_like(mappa)))
        new_diagonal = np.zeros_like(mappa)
        new_diagonal[np.where(diagonal == 1)] = 1
        ax.imshow(new_diagonal, cmap="binary", vmin=0, vmax=1, alpha=new_diagonal, interpolation="nearest")
    pp.make_alphacolorbar(cax,vmin_rise,vmax_rise,0.5,alphamin,alphamax,2,cmap=cmap,around=1,lbl_g=False,alphaticklabels=alphaticklabels)
    cax.set_xlabel(alphalbl,fontsize=15)
    if use_rise_times:
        cax.set_ylabel(r'$\langle$rise time (s)$\rangle$',fontsize=15)
    else:
        cax.set_ylabel(r'$\langle$decay time (s)$\rangle$',fontsize=15)
    cax.tick_params(labelsize=15)
    ax.set_xlabel("Stimulated",fontsize=30)
    ax.set_ylabel("Responding",fontsize=30)
    ax.set_xticks(np.arange(len(sorter_j)))
    ax.set_yticks(np.arange(len(sorter_i)))
    if SIM:
        ax.set_xticklabels(funa.SIM_head_ids[sorter_j],fontsize=5,rotation=90)
        ax.set_yticklabels(funa.SIM_head_ids[sorter_i],fontsize=5)
    else:
        ax.set_xticklabels(funa.head_ids[sorter_j],fontsize=5,rotation=90)
        ax.set_yticklabels(funa.head_ids[sorter_i],fontsize=5)
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
    ax.set_xlim(-0.5,lim)
    if stamp: pp.provstamp(ax,-.1,-.1," ".join(sys.argv))

if plot: plt.tight_layout()
folder='/projects/LEIFER/francesco/funatlas/'
txt_fname = "funatlas_timescales_map.txt"
if save:
    #folder='/home/leifer/Desktop/fig_transitory/'
    if plot:
        plt.savefig(folder+"funatlas_timescales_map.png", dpi=300, bbox_inches="tight",metadata={"Comment":" ".join(sys.argv)})
        plt.savefig(folder+"funatlas_timescales_map.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(folder+"funatlas_timescales_map.svg", bbox_inches="tight")
    if len(ds_exclude_i)>0: 
        txt_fname = "funatlas_timescales_map_excl_"+"-".join([str(a) for a in ds_exclude_i])+".txt"
    np.savetxt(folder+txt_fname,mappa_full)
    if to_paper and plot:
        plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS8/funatlas_timescales_map.pdf", dpi=600, bbox_inches="tight")

plt.show()
