import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": True}
                 

# Load Funatlas for actual data
funa_wt = pp.Funatlas.from_datasets(ds_list,merge_bilateral=False,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=True,
                                 ds_tags=None,ds_exclude_tags="mutant",
                                 verbose=False)
                                 
# Load Funatlas for actual data
funa_unc31 = pp.Funatlas.from_datasets(ds_list,merge_bilateral=False,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=True,
                                 ds_tags="unc31",ds_exclude_tags=None,
                                 verbose=False)
                                 
_,inclall_occ2_wt = funa_wt.get_occurrence_matrix(req_auto_response=True,inclall=True)
_,inclall_occ2_unc31 = funa_unc31.get_occurrence_matrix(req_auto_response=True,inclall=True)
                                 
q_wt =  funa_wt.get_kolmogorov_smirnov_q(inclall_occ2_wt)
q_unc31 =  funa_unc31.get_kolmogorov_smirnov_q(inclall_occ2_unc31,strain="unc31") 

fig = plt.figure(1,figsize=(4,3))
ax = fig.add_subplot(111)
_,bins,_ = ax.hist(q_unc31[~np.isnan(q_wt)],bins=50,alpha=0.3,density=True,label="unc-31",color="C1")
ax.hist(q_wt[~np.isnan(q_wt)],bins=bins,alpha=0.3,density=True,label="WT",color="C0")
ax.axvline(0.05,c="k",ls="--")
ax.set_xticks([0.,0.4,0.8])
ax.set_yticks([1,2,3,4])
ax.set_xlabel("q value")
ax.set_ylabel("density")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.legend()

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/qvalues_wt_unc31.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/qvalues_wt_unc31.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/qvalues_wt_unc31.pdf",bbox_inches="tight")

fig = plt.figure(2,figsize=(4,3))
ax = fig.add_subplot(111)
a = np.sum(q_wt<0.05) / np.sum(~np.isnan(q_wt))
b = np.sum(q_unc31<0.05) / np.sum(~np.isnan(q_unc31))
x = np.arange(2)*0.05
ax.bar(0,a,width=0.4,align="center",color="C0")
ax.bar(0.5,b,width=0.4,align="center",color="C1")
ax.set_xticks([0,0.5])
ax.set_xticklabels(["WT","unc-31"])
ax.set_ylabel("frac. of pairs with\nq<0.05 connections")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/qvalues_wt_unc31_bars.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/qvalues_wt_unc31_bars.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/qvalues_wt_unc31_bars.pdf",bbox_inches="tight")

plt.show()
