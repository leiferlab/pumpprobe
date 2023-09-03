import numpy as np, matplotlib.pyplot as plt, sys
from scipy.stats import kstest
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

matchless_nan_th = 0.5
matchless_nan_th_added_only = True
matchless_nan_th_from_file = False

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}
                 
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

# Occurrence matrix
_,occ2_wt = funa_wt.get_occurrence_matrix(req_auto_response=True)
_,occ2_unc31 = funa_unc31.get_occurrence_matrix(req_auto_response=True)

# Get kernel rise times
time1 = np.linspace(0,30,1000)
time2 = np.linspace(0,200,1000)

rise_times_wt = funa_wt.get_eff_rise_times(occ2_wt,time2,False,drop_saturation_branches=False)
rise_times_unc31 = funa_unc31.get_eff_rise_times(occ2_unc31,time2,False,drop_saturation_branches=False)
decay_times_wt = funa_wt.get_eff_decay_times(occ2_wt,time2,False,drop_saturation_branches=False)
decay_times_unc31 = funa_unc31.get_eff_decay_times(occ2_unc31,time2,False,drop_saturation_branches=False)

# Take average rise time weighted by dFF and label confidence
dFF_wt = funa_wt.get_max_deltaFoverF(occ2_wt,time1)
avg_rise_times_wt = funa_wt.weighted_avg_occ2style2(rise_times_wt,[dFF_wt])
avg_decay_times_wt = funa_wt.weighted_avg_occ2style2(decay_times_wt,[dFF_wt])

dFF_unc31 = funa_unc31.get_max_deltaFoverF(occ2_unc31,time1)
avg_rise_times_unc31 = funa_unc31.weighted_avg_occ2style2(rise_times_unc31,[dFF_unc31])
avg_decay_times_unc31 = funa_unc31.weighted_avg_occ2style2(decay_times_unc31,[dFF_unc31])

rel_rise_change = (avg_rise_times_unc31-avg_rise_times_wt)/avg_rise_times_wt
rel_decay_change = (avg_decay_times_unc31-avg_decay_times_wt)/avg_decay_times_wt

diag_i = np.diag_indices(rel_rise_change.shape[0])

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.hist(rel_rise_change[diag_i],bins=30)
ax.axvline(np.nanmedian(rel_rise_change[diag_i]),c="k")
ax.set_xlabel("Relative change in rise-time of the autoresponse unc-31 - WT")
ax.set_ylabel("n")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/reviewers/timescales_rel_change_rise.png",dpi=300,bbox_inches="tight")

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.hist(rel_decay_change[diag_i],bins=30)
ax.axvline(np.nanmedian(rel_decay_change[diag_i]),c="k")
ax.set_xlabel("Relative change in decay-time of the autoresponse unc-31 - WT")
ax.set_ylabel("n")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/reviewers/timescales_rel_change_decay.png",dpi=300,bbox_inches="tight")

plt.show()


