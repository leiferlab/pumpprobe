import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

q_th = 0.05
leq_rise_time = 0.02

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
funa_wt = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=True,
                                 ds_tags=None,ds_exclude_tags="mutant",
                                 verbose=False)
                                 
# Load Funatlas for actual data
funa_unc31 = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=True,
                                 ds_tags="unc31",ds_exclude_tags=None,
                                 verbose=False)

# Occurrence matrix
_,occ2_wt = funa_wt.get_occurrence_matrix(req_auto_response=True)
_,occ2_unc31 = funa_unc31.get_occurrence_matrix(req_auto_response=True)
_,inclall_occ2_wt = funa_wt.get_occurrence_matrix(req_auto_response=True,inclall=True)
_,inclall_occ2_unc31 = funa_unc31.get_occurrence_matrix(req_auto_response=True,inclall=True)

# q values                                 
q_wt =  funa_wt.get_kolmogorov_smirnov_q(inclall_occ2_wt)
q_unc31 =  funa_unc31.get_kolmogorov_smirnov_q(inclall_occ2_unc31,strain="unc31")

# Get kernel rise times
time1 = np.linspace(0,30,1000)
time2 = np.linspace(0,200,1000)

rise_times_wt = funa_wt.get_eff_rise_times(occ2_wt,time2,True,drop_saturation_branches=True)
rise_times_unc31 = funa_unc31.get_eff_rise_times(occ2_unc31,time2,True,drop_saturation_branches=True)

# Take average rise time weighted by dFF and label confidence
_, conf2_wt = funa_wt.get_labels_confidences(occ2_wt)
dFF_wt = funa_wt.get_max_deltaFoverF(occ2_wt,time1)
avg_rise_times_wt = funa_wt.weighted_avg_occ2style2(rise_times_wt,[dFF_wt,conf2_wt])

_, conf2_unc31 = funa_unc31.get_labels_confidences(occ2_unc31)
dFF_unc31 = funa_unc31.get_max_deltaFoverF(occ2_unc31,time1)
avg_rise_times_unc31 = funa_unc31.weighted_avg_occ2style2(rise_times_unc31,[dFF_unc31,conf2_unc31])

# Make matrices for fast and slow connections
occ1fastbool_wt = avg_rise_times_wt<=leq_rise_time
occ1slowbool_wt = avg_rise_times_wt>leq_rise_time

occ1fastbool_unc31 = avg_rise_times_unc31<=leq_rise_time
occ1slowbool_unc31 = avg_rise_times_unc31>leq_rise_time

# Compute fraction of q<q_th connections that are fast (nans should be gone with the inequality above)
fast_wt = np.sum(occ1fastbool_wt)/np.sum(q_wt<q_th)
fast_unc31 = np.sum(occ1fastbool_unc31)/np.sum(q_unc31<q_th)

fig = plt.figure(1,figsize=(4,6))
ax = fig.add_subplot(111)
ax.bar((0),(fast_wt),color="C0")
ax.bar((1),(fast_unc31),color="C1")
ax.set_xticks([0,1])
ax.set_xticklabels(["WT","unc-31"])
ax.set_yticks([0.,0.2,0.4])
ax.set_ylabel("Frac of q<"+str(q_th)+"\nthat are fast")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig.tight_layout()
#fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/fraction_that_is_fast_wt_unc31.png",dpi=300,bbox_inches="tight")
#fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/fraction_that_is_fast_wt_unc31.png",dpi=300,bbox_inches="tight")
#fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/fraction_that_is_fast_wt_unc31.pdf",bbox_inches="tight")
plt.show()

'''
# Load activity connectome (anatomy-derived effective weights)
act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_bilateral_merged.txt")
act_conn[np.isnan(act_conn)] = 0

# Sparsify the activity connectome to match the sparseness of the q<q_th matrix
sparseness_wt = np.sum(q_wt<q_th)/np.prod(q_wt.shape)
th_wt = pp.Funatlas.threshold_to_sparseness(act_conn,sparseness_wt)
act_conn_wt = np.abs(act_conn)>th_wt

sparseness_unc31 = np.sum(q_unc31<q_th)/np.prod(q_unc31.shape)
th_unc31 = pp.Funatlas.threshold_to_sparseness(act_conn,sparseness_unc31)
act_conn_unc31 = np.abs(act_conn)>th_unc31'''



