import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm

plt.rc("xtick",labelsize=14)
plt.rc("ytick",labelsize=14)

sort = "--sort" in sys.argv
pop_nans = "--pop-nans" in sys.argv
rele = "--rele" in sys.argv # Plot relative errors
rele_suff = "" if not rele else "_rele"
rele_label = "(s)" if not rele else "rel. err."

alpha_qvalues = "--alpha-qvalues" in sys.argv
alpha_occ1 = not alpha_qvalues

use_kernels = "--use-kernels" in sys.argv
drop_saturation_branches = "--drop-saturation-branches" in sys.argv

stamp = "--no-stamp" not in sys.argv

q_th = 0.05
print("thresholding by qvalue",q_th)

# vmin and vmax for the different plots
if not use_kernels:
    vmax_rise = 30.0 if not rele else 2.0
    vmin_rise = 5 if not rele else 0
    vmax_decay = 60.0 if not rele else 2.0
    vmin_decay = 10 if not rele else 0
    vmax_peak = 30.0 if not rele else 2.0
    vmin_peak = 5 if not rele else 0
else:
    vmax_rise = 2.0 if not rele else 2.0
    vmin_rise = 0 if not rele else 0
    vmax_decay = 5.0 if not rele else 2.0
    vmin_decay = 0 if not rele else 0
    vmax_peak = 5.0 if not rele else 2.0
    vmin_peak = 0 if not rele else 0

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="D20 E32 old mutant",#None"mutant",
                verbose=False)
                
occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=use_kernels)
occ3 = funa.get_observation_matrix()

# FILTER BY Q VALUES
_,occ2_inclall = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
qvalues,pvalues = funa.get_kolmogorov_smirnov_q(occ2_inclall,return_p=True)
qvalues[np.isnan(qvalues)] = 1.
for ii in np.arange(qvalues.shape[0]):
    for jj in np.arange(qvalues.shape[1]):
        if qvalues[ii,jj]>q_th:
            occ2[ii,jj] = []
occ1 = funa.regenerate_occ1(occ2)

conf1, conf2 = funa.get_labels_confidences(occ2)

# Get timescales
if not use_kernels:
    time1 = np.linspace(0,30,1000)
    time2 = np.linspace(0,200,1000)
else:
    time1 = np.linspace(0,30,1000)
    time2 = np.linspace(0,10,1000)
dFF = funa.get_max_deltaFoverF(occ2,time1)
rise_times = funa.get_eff_rise_times(occ2,time2,use_kernels,drop_saturation_branches)
decay_times = funa.get_eff_decay_times(occ2,time2,use_kernels,drop_saturation_branches)
peak_times = funa.get_peak_times(occ2,time2,use_kernels,drop_saturation_branches)

# Taking the weighted average with weights dFF*conf
avg_rise_times, rele_rise_times = funa.weighted_avg_occ2style2(rise_times,[dFF,conf2],return_rele=True)
avg_decay_times, rele_decay_times = funa.weighted_avg_occ2style2(decay_times,[dFF,conf2],return_rele=True)
avg_peak_times, rele_peak_times = funa.weighted_avg_occ2style2(peak_times,[dFF,conf2],return_rele=True)

# Setting the transparency of the imshows
if alpha_occ1:
    occ1_head = funa.reduce_to_head(occ1)
    occ1_head = occ1_head/np.max(occ1_head)
    alphalbl = "n responses\n(norm to max)"
    alphamin,alphamax = 0.,0.1#0.25, 1
    alphas = np.clip((occ1_head-alphamin)/(alphamax-alphamin),0,1)
elif alpha_qvalues:
    qvalues = funa.get_qvalues(occ1,occ3,False)
    qvalues_head = funa.reduce_to_head(qvalues)
    qvalues_head[np.isnan(qvalues_head)] = 1.
    alphas = (1.-qvalues_head)
    alphalbl = "1-Q"
    alphamin,alphamax = 0.,0.9
    alphas = np.clip((alphas-alphamin)/(alphamax-alphamin),0,1)
alphacbarstep = 1 if use_kernels else 5

# Plotting the rise times
mappa = avg_rise_times if not rele else rele_rise_times
mappa = funa.reduce_to_head(mappa)
if sort:
    mappa,sorter,lim = funa.sort_matrix_nans(mappa,axis=-1,return_all=True)
    sorter_i = sorter_j = sorter
    alphas = alphas[sorter][:,sorter]
elif pop_nans:
    mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(mappa,return_all=True)
    alphas = alphas[sorter_i][:,sorter_j]
else:
    sorter_i = sorter_j = np.arange(mappa.shape[-1])
    lim = None
    
fig1 = plt.figure(1,figsize=(10.5,10))
gs = fig1.add_gridspec(1,10)
ax = fig1.add_subplot(gs[0,:9])
cax = fig1.add_subplot(gs[0,9:])
#new
ax.imshow(0.*np.ones_like(mappa),cmap="Greys",vmax=1,vmin=0)
blank_mappa = np.copy(mappa)
blank_mappa[~np.isnan(mappa)] = 0.1
ax.imshow(blank_mappa,cmap="Greys",vmin=0,vmax=1)
#new
im = ax.imshow(mappa,cmap="viridis",vmin=vmin_rise,vmax=vmax_rise,alpha=alphas)
pp.make_alphacolorbar(cax,vmin_rise,vmax_rise,alphacbarstep,alphamin,alphamax,2)
cax.set_xlabel(alphalbl,fontsize=15)
cax.set_ylabel("rise time (s)",fontsize=15)
cax.tick_params(labelsize=15)
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(sorter_j)))
ax.set_yticks(np.arange(len(sorter_i)))
ax.set_xticklabels(funa.head_ids[sorter_j],fontsize=6,rotation=90)
ax.set_yticklabels(funa.head_ids[sorter_i],fontsize=6)
ax.set_xlim(-0.5,lim)
if use_kernels: ax.set_title("kernels")
if stamp: pp.provstamp(ax,-.1,-.05,"".join(sys.argv))
fig1.tight_layout()
fig1.savefig("/projects/LEIFER/francesco/funatlas/funatlas_rise_time_map"+rele_suff+".png", dpi=300, bbox_inches="tight",metadata={"Comment":" ".join(sys.argv)})
fig1.savefig("/projects/LEIFER/francesco/funatlas/funatlas_rise_time_map"+rele_suff+".pdf", dpi=300, bbox_inches="tight")
fig1.savefig("/projects/LEIFER/francesco/funatlas/funatlas_rise_time_map"+rele_suff+".svg", dpi=300, bbox_inches="tight")

fig4 = plt.figure(4,figsize=(5,3))
ax = fig4.add_subplot(111)
rise_times_rav = []
for rt in np.ravel(rise_times):
    if rt is not None:
        for rt_ in rt: rise_times_rav.append(rt_)
ax.hist(np.array(rise_times_rav),bins=100,range=(0,vmax_rise),density=True)
ax.hist(np.ravel(avg_rise_times),bins=100,range=(0,vmax_rise),density=True)
ax.set_xlabel("rise time (s)",fontsize=18)
ax.set_ylabel("density",fontsize=18)
#if use_kernels: ax.set_title("kernels")
if stamp: pp.provstamp(ax,-.1,-.05,"".join(sys.argv))
fig4.tight_layout()
fig4.savefig("/home/frandi/rise_times_qth_"+str(q_th)+".png",dpi=300,bbox_inches="tight")

# Plotting the decay times
mappa = avg_decay_times if not rele else rele_decay_times
mappa = funa.reduce_to_head(mappa)
if sort:
    mappa,sorter,lim = funa.sort_matrix_nans(mappa,axis=-1,return_all=True)
    sorter_i = sorter_j = sorter
    alphas = alphas[sorter][:,sorter]
elif pop_nans:
    mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(mappa,return_all=True)
else:
    sorter_i = sorter_j = np.arange(mappa.shape[-1])
    lim = None

fig2 = plt.figure(2,figsize=(10.5,10))
gs = fig2.add_gridspec(1,10)
ax = fig2.add_subplot(gs[0,:9])
cax = fig2.add_subplot(gs[0,9:])
#new
ax.imshow(0.*np.ones_like(mappa),cmap="Greys",vmax=1,vmin=0)
blank_mappa = np.copy(mappa)
blank_mappa[~np.isnan(mappa)] = 0.1
ax.imshow(blank_mappa,cmap="Greys",vmin=0,vmax=1)
#new
im = ax.imshow(mappa,cmap="viridis",vmin=vmin_decay,vmax=vmax_decay,alpha=alphas)
pp.make_alphacolorbar(cax,vmin_decay,vmax_decay,alphacbarstep,alphamin,alphamax,2)
cax.set_xlabel(alphalbl,fontsize=15)
cax.set_ylabel("decay times "+rele_label,size=15)
cax.tick_params(labelsize=15)
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(sorter_j)))
ax.set_yticks(np.arange(len(sorter_i)))
ax.set_xticklabels(funa.head_ids[sorter_j],fontsize=5,rotation=90)
ax.set_yticklabels(funa.head_ids[sorter_i],fontsize=5)
ax.set_xlim(-0.5,lim)
if use_kernels: ax.set_title("kernels")
if stamp: pp.provstamp(ax,-.1,-.05,"".join(sys.argv))
fig2.tight_layout()
fig2.savefig("/projects/LEIFER/francesco/funatlas/funatlas_decay_time_map"+rele_suff+".png", dpi=300, bbox_inches="tight",metadata={"Comment":" ".join(sys.argv)})
fig2.savefig("/projects/LEIFER/francesco/funatlas/funatlas_decay_time_map"+rele_suff+".pdf", dpi=300, bbox_inches="tight")
fig2.savefig("/projects/LEIFER/francesco/funatlas/funatlas_decay_time_map"+rele_suff+".svg", dpi=300, bbox_inches="tight")

fig5 = plt.figure(5)
ax = fig5.add_subplot(111)
decay_times_rav = []
for dt in np.ravel(decay_times):
    if dt is not None:
        for dt_ in dt: decay_times_rav.append(dt_)
ax.hist(np.array(decay_times_rav),bins=100,range=(0,vmax_decay),density=True)
ax.set_xlabel("decay time (s)",fontsize=14)
ax.set_ylabel("density",fontsize=14)
if use_kernels: ax.set_title("kernels")
if stamp: pp.provstamp(ax,-.1,-.05,"".join(sys.argv))
fig5.tight_layout()

mappa = avg_peak_times if not rele else rele_peak_times
mappa = funa.reduce_to_head(mappa)
if sort:
    mappa,sorter,lim = funa.sort_matrix_nans(mappa,axis=-1,return_all=True)
    sorter_i = sorter_j = sorter
    alphas = alphas[sorter][:,sorter]
elif pop_nans:
    mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(mappa,return_all=True)
else:
    sorter_i = sorter_j = np.arange(mappa.shape[-1])
    lim = None
    

# Plotting the peak times
fig3 = plt.figure(3,figsize=(10.5,10))
gs = fig3.add_gridspec(1,10)
ax = fig3.add_subplot(gs[0,:9])
cax = fig3.add_subplot(gs[0,9:])
#new
ax.imshow(0.*np.ones_like(mappa),cmap="Greys",vmax=1,vmin=0)
blank_mappa = np.copy(mappa)
blank_mappa[~np.isnan(mappa)] = 0.1
ax.imshow(blank_mappa,cmap="Greys",vmin=0,vmax=1)
#new
im = ax.imshow(mappa,cmap="viridis",vmin=vmin_peak,vmax=vmax_peak,alpha=alphas)
pp.make_alphacolorbar(cax,vmin_peak,vmax_peak,alphacbarstep,alphamin,alphamax,2)
cax.set_xlabel(alphalbl,fontsize=15)
cax.set_ylabel("peak times "+rele_label,size=15)
cax.tick_params(labelsize=15)
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(sorter_j)))
ax.set_yticks(np.arange(len(sorter_i)))
ax.set_xticklabels(funa.head_ids[sorter_j],fontsize=5,rotation=90)
ax.set_yticklabels(funa.head_ids[sorter_i],fontsize=5)
ax.set_xlim(-0.5,lim)
if use_kernels: ax.set_title("kernels")
if stamp: pp.provstamp(ax,-.1,-.05,"".join(sys.argv))
fig3.tight_layout()
fig3.savefig("/projects/LEIFER/francesco/funatlas/funatlas_peak_time_map"+rele_suff+".png", dpi=300, bbox_inches="tight",metadata={"Comment":" ".join(sys.argv)})
fig3.savefig("/projects/LEIFER/francesco/funatlas/funatlas_peak_time_map"+rele_suff+".pdf", dpi=300, bbox_inches="tight")
fig3.savefig("/projects/LEIFER/francesco/funatlas/funatlas_peak_time_map"+rele_suff+".svg", dpi=300, bbox_inches="tight")

fig6 = plt.figure(6)
ax = fig6.add_subplot(111)
peak_times_rav = []
for pt in np.ravel(peak_times):
    if pt is not None:
        for pt_ in pt: peak_times_rav.append(pt_)
ax.hist(np.array(peak_times_rav),bins=100,range=(0,vmax_peak),density=True)
ax.set_xlabel("peak time (s)",fontsize=14)
ax.set_ylabel("density",fontsize=14)
if use_kernels: ax.set_title("kernels")
if stamp: pp.provstamp(ax,-.1,-.05,"".join(sys.argv))
fig6.tight_layout()


plt.show()
