import numpy as np, matplotlib.pyplot as plt
import os, sys, json
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

merge = "--no-merge" not in sys.argv
print("Merge:",merge)
unc31 = "--unc31" in sys.argv
strain = "" if not unc31 else "unc31"
ds_tags = None if not unc31 else "unc31"
ds_exclude_tags = "mutant" if not unc31 else None
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv

for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])

acfolder = "/projects/LEIFER/francesco/simulations/activity_connectome_sign2/"
if not merge:
    actconncorr = np.loadtxt(acfolder+"activity_connectome_correlation_no_merge.txt")
else:
    actconncorr = np.loadtxt(acfolder+"activity_connectome_bilateral_merged_correlation.txt")

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_list_spont = "/projects/LEIFER/francesco/funatlas_list_spont.txt"
                  
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=merge,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=False,
                                 ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                                 verbose=False)
                                 
funa_spont = pp.Funatlas.from_datasets(ds_list_spont, merge_bilateral=merge,
                                         signal="green",
                                         signal_kwargs = signal_kwargs)

occ1,occ2 = funa.get_occurrence_matrix(req_auto_response=True)
occ3 = funa.get_observation_matrix(req_auto_response=True)
_,inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
q,p = funa.get_kolmogorov_smirnov_q(inclall_occ2,return_p=True,strain=strain)

# Get the kernel-derived correlations
km = funa.get_kernels_map(occ2,occ3,filtered=True,include_flat_kernels=True)
km[q>0.05]=np.nan
ck = funa.get_correlation_from_kernels_map(km,occ3,set_unknown_to_zero=False)
ck[q>0.05]=np.nan
nanmask = np.isnan(ck)*np.isnan(ck.T)
ck = 0.5*(np.nansum([ck,ck.T],axis=0))
ck[nanmask] = np.nan

# Get the stimulated activity correlation
stimcorr = funa.get_signal_correlations()

# Get the spontaneous activity correlation
spontcorr = funa_spont.get_signal_correlations()

# Get the direct anatomical connectome
aconn = funa.aconn_chem + funa.aconn_gap

# Get the effective anatomical connectome
Aconn = funa.get_effective_aconn4(gain_1=646.4)
Aconnsym = funa.corr_from_eff_causal_conn(Aconn)

# Resymmetrize actconcorr
nanmask = np.isnan(actconncorr)*np.isnan(actconncorr.T)
actconncorr = 0.5*(np.nansum([actconncorr,actconncorr.T],axis=0))
actconncorr[nanmask] = np.nan

# Prepare array to exclude elements on the diagonal
ondiag = np.zeros_like(q,dtype=bool)
np.fill_diagonal(ondiag,True)

excl = np.isnan(spontcorr) | np.isnan(ck) | np.isnan(actconncorr) | ondiag | np.isnan(stimcorr)
np.savetxt("/projects/LEIFER/francesco/funatlas/excl_vs_correlations.txt",excl)
excl_min = np.isnan(spontcorr) | ondiag

############################################################
# COMPUTE CORRELATION COEFFICIENTS WITH SPONTANEOUS ACTIVITY
############################################################
#spontcorr_ = spontcorr[~excl]

# Anatomical connectome (bare)
excl = excl_min
spontcorr_ = spontcorr[~excl]
aconn_ = aconn[~excl]
r_spontcorr_aconn = np.corrcoef([spontcorr_,aconn_])[0,1]

# Effective anatomical connectome (linear)
excl = excl_min
spontcorr_ = spontcorr[~excl]
Aconnsym_ = Aconnsym[~excl]
r_spontcorr_Aconnsym = np.corrcoef([spontcorr_,Aconnsym_])[0,1]

# Kernel-derived correlations (all driven)
excl = excl_min | np.isnan(ck)
spontcorr_ = spontcorr[~excl]
ck_ = ck[~excl]
r_spontcorr_ck = np.corrcoef([spontcorr_,ck_])[0,1]

# Anatomy-derived correlations (biophysical, all driven)
excl = excl_min
spontcorr_ = spontcorr[~excl]
actconncorr_ = actconncorr[~excl]
r_spontcorr_actconncorr = np.corrcoef([spontcorr_,actconncorr_])[0,1]

# Stimulated-activity correlations
excl = excl_min | np.isnan(stimcorr)
spontcorr_ = spontcorr[~excl]
stimcorr_ = stimcorr[~excl]
r_spontcorr_stimcorr = np.corrcoef([spontcorr_,stimcorr_])[0,1]

###############
# PRINT RESULTS
###############

print("TAKE THESE TWO VALUES FOR THE BAR PLOT funatlas_vs_correlation_combined_bar_plot.py")
print('''anatomical connectome vs spontaneous correlations''', r_spontcorr_aconn)
print('''connectome-derive correlation (linear) vs spontaneous correlations''', r_spontcorr_Aconnsym)
print('''connectome-derived correlation (biophysical) vs spontaneous correlations''',r_spontcorr_actconncorr)
print('''kernel-derived correlation vs spontaneous correlations''',r_spontcorr_ck)
print('''stimulated correlation vs spontaneous correlations''',r_spontcorr_stimcorr)


