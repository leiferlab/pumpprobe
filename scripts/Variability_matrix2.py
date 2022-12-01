import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
from multipy import fdr
from scipy import stats
import cmasher as cmr

#this script is to makes the plots for the observation number, the variability in kernels and the fraction responding
SIM = "--sim" in sys.argv
pop_nans = "--pop-nans" in sys.argv
stamp = "--no-stamp" not in sys.argv
ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "mutant"
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
two_min_occ = "--two-min-occ" in sys.argv
figsize = (12,10)
enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv

for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])

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

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ,req_auto_response=req_auto_response)
# If occ2 needs to be filtered
#occ1,occ2 = funa.filter_occ12_from_sysargv(occ2,sys.argv)

#this gets the version of occ3 that has only the traces that pass the nan threshold
occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=req_auto_response)

inclall_occ2 = occ2


#occ3_head = funa.reduce_to_head(occ3)


iids = jids = funa.neuron_ids  #gives ids for each global index
#next calculate the variability matrix
variability_matrix = np.zeros((funa.n_neurons, funa.n_neurons)) * np.nan

for ai in np.arange(funa.n_neurons):
    for aj in np.arange(funa.n_neurons):

        if occ3[ai, aj] > 1:
            # collectiong all the kernels and stimulations
            current_kernels = []
            current_stimuli = []
            for occ in occ2[ai, aj]:  # for each of the response and stimulus instances
                ds = occ["ds"]  # dataset number
                ie = occ["stim"]  # stimulation number
                resp_i = occ["resp_neu_i"]  # number of the responding neuron
                # Build the time axis
                i0 = funa.fconn[ds].i0s[ie]
                i1 = funa.fconn[ds].i1s[ie]
                Dt = funa.fconn[ds].Dt
                next_stim_in_how_many = funa.fconn[ds].next_stim_after_n_vol
                shift_vol = funa.fconn[ds].shift_vol
                time_trace = (np.arange(i1 - i0) - shift_vol) * Dt
                time = (np.arange(62)) * Dt  # this is the time axis starting from 0, from the time_0_vol

                stim_j = funa.fconn[ds].stim_neurons[ie]
                # only if there are kernels
                if not np.array_equal(funa.fconn[ds].fit_params[ie][resp_i]['params'],
                                      funa.fconn[ds].fit_params_default['params']) or not \
                funa.fconn[ds].fit_params[ie][resp_i]['n_branches'] == funa.fconn[ds].fit_params_default[
                    'n_branches'] or not np.array_equal(funa.fconn[ds].fit_params[ie][resp_i]['n_branch_params'],
                                                        funa.fconn[ds].fit_params_default['n_branch_params']):

                    # Generate the fitted stimulus activity
                    stim_unc_par = funa.fconn[ds].fit_params_unc[ie][stim_j]  #
                    if stim_unc_par["n_branches"] == 0:
                        print("skipping " + str(ie))
                        continue

                    else:

                        # y = funatlas.fconn[ds].get_kernel(time, ie, resp_i)
                        ker_ec = funa.fconn[ds].get_kernel_ec(ie, resp_i)
                        stim_ec = pp.ExponentialConvolution.from_dict(stim_unc_par)

                        current_kernels.append(ker_ec)
                        current_stimuli.append(stim_ec)

            # for every pair we go through all the pairwise kernels to calculate the correlations
            current_correlations = []
            for k in np.arange(len(current_kernels)):
                for k2 in range(k+1, len(current_kernels)):
                    if k != k2:
                        ec1 = current_stimuli[k]
                        ec2 = current_stimuli[k2]

                        fit_stim1 = ec1.eval(time)
                        fit_stim2 = ec2.eval(time)

                        kernel1 = current_kernels[k].eval(time)
                        kernel2 = current_kernels[k2].eval(time)

                        response11 = pp.convolution(fit_stim1, kernel1, Dt, 8)
                        response12 = pp.convolution(fit_stim1, kernel2, Dt, 8)
                        try:
                            corr_coef_pair1, _ = stats.pearsonr(response11, response12)
                        except:
                            print(response11)

                        response21 = pp.convolution(fit_stim2, kernel1, Dt, 8)
                        response22 = pp.convolution(fit_stim2, kernel2, Dt, 8)
                        corr_coef_pair2, _ = stats.pearsonr(response21, response22)

                        current_correlations.append(corr_coef_pair1)
                        current_correlations.append(corr_coef_pair2)
            # take the average of all the correlations in a pair and add that to the variability matrix
            average_correlation = np.mean(current_correlations)
            variability_matrix[ai,aj] = average_correlation

if SIM:
    variability_matrix_head = funa.reduce_to_SIM_head(variability_matrix)
else:
    variability_matrix_head = funa.reduce_to_head(variability_matrix)


#######################################################################################################
# First we make the Occ1/Occ3 figure and calculate the sorters for Occ3 to use for the rest of the axes

occ1_notinclall, occ2_notinclall = funa.get_occurrence_matrix(inclall=False,req_auto_response=req_auto_response)
cmap = cmr.get_sub_cmap('Purples', 0.2, 1.0)
fraction_respond = occ1_notinclall/occ3
if SIM:
    fraction_respond_head = funa.reduce_to_SIM_head(fraction_respond)
else:
    fraction_respond_head = funa.reduce_to_head(fraction_respond)

if pop_nans:
    if SIM:
        fraction_respond_head, sorter_i, sorter_j, lim = funa.sort_matrix_pop_nans_SIM(fraction_respond_head)
    else:
        fraction_respond_head,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(fraction_respond_head,return_all=True)
else:
    sorter_i = sorter_j = np.arange(fraction_respond_head.shape[-1])
    lim = None
fig2 = plt.figure(1, figsize=figsize)
ax2 = fig2.add_subplot()
ax2.imshow(0. * np.ones_like(fraction_respond_head), cmap="Greys", vmax=1, vmin=0)
#blank_mappa = np.copy(fraction_respond_head)
#blank_mappa[~np.isnan(fraction_respond_head)] = 0.2
#ax2.imshow(blank_mappa, cmap="Greys", vmin=0, vmax=1)
im = ax2.imshow(fraction_respond_head, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
diagonal = np.diag(np.diag(np.ones_like(fraction_respond_head)))
new_diagonal = np.zeros_like(fraction_respond_head)
new_diagonal[np.where(diagonal == 1)] = 1
ax2.imshow(new_diagonal, cmap="binary", vmin=0, vmax=1, alpha=new_diagonal, interpolation="nearest")
plt.colorbar(im, label = r'Fraction Responding')
ax2.set_xlabel("stimulated",fontsize=30)
ax2.set_ylabel("responding",fontsize=30)
ax2.set_xticks(np.arange(len(sorter_j)))
ax2.set_yticks(np.arange(len(sorter_i)))
if SIM:
    ax2.set_xticklabels(funa.SIM_head_ids[sorter_j], fontsize=4, rotation=90)
    ax2.set_yticklabels(funa.SIM_head_ids[sorter_i], fontsize=4)
else:
    ax2.set_xticklabels(funa.head_ids[sorter_j], fontsize=4, rotation=90)
    ax2.set_yticklabels(funa.head_ids[sorter_i], fontsize=4)
ax2.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax2.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
ax2.set_xlim(-0.5, lim)
if merge_bilateral:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/fraction_responding_merged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/fraction_responding_merged.pdf")
else:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/fraction_responding_unmerged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/fraction_responding_unmerged.pdf")
fig2.clf()

if SIM:
    occ3_head = funa.reduce_to_SIM_head(occ3)
else:
    occ3_head = funa.reduce_to_head(occ3)
occ3_head = occ3_head[sorter_i][:,sorter_j]

cmap = "binary"
fig3 = plt.figure(1, figsize=figsize)
ax3 = fig3.add_subplot()
#ax3.imshow(0. * np.ones_like(occ3_head), cmap="Greys", vmax=1, vmin=0)
#blank_mappa = np.copy(occ3_head)
#blank_mappa[~np.isnan(occ3_head)] = 0.2
#ax3.imshow(blank_mappa, cmap="Greys", vmin=0, vmax=1)
im = ax3.imshow(occ3_head, cmap=cmap, vmin=0, vmax=np.nanmax(occ3_head), interpolation="nearest")
plt.colorbar(im, label = r'Number of Observations')
ax3.set_xlabel("stimulated",fontsize=30)
ax3.set_ylabel("responding",fontsize=30)
ax3.set_xticks(np.arange(len(sorter_j)))
ax3.set_yticks(np.arange(len(sorter_i)))
if SIM:
    ax3.set_xticklabels(funa.SIM_head_ids[sorter_j], fontsize=4, rotation=90)
    ax3.set_yticklabels(funa.SIM_head_ids[sorter_i], fontsize=4)
else:
    ax3.set_xticklabels(funa.head_ids[sorter_j], fontsize=4, rotation=90)
    ax3.set_yticklabels(funa.head_ids[sorter_i], fontsize=4)
ax3.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax3.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
ax3.set_xlim(-0.5, lim)
if merge_bilateral:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/occ3_merged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/occ3_merged.pdf")
else:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/occ3_unmerged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/occ3_unmerged.pdf")
fig3.clf()

variability_matrix_head = variability_matrix_head[sorter_i][:,sorter_j]
vmax = 1
cmap = cmr.get_sub_cmap('Greens', 0.2, 1.0)

fig1 = plt.figure(1, figsize=figsize)
ax = fig1.add_subplot()
ax.imshow(0. * np.ones_like(variability_matrix_head), cmap="Greys", vmax=1, vmin=0)
blank_mappa = np.copy(variability_matrix_head)
blank_mappa[~np.isnan(variability_matrix_head)] = 0.1
ax.imshow(blank_mappa, cmap="Greys", vmin=0, vmax=1)
im = ax.imshow(variability_matrix_head, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.imshow(new_diagonal, cmap="binary", vmin=0, vmax=1, alpha=new_diagonal, interpolation="nearest")
plt.colorbar(im, label = r'Ave Correlation Coefficient')
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(sorter_j)))
ax.set_yticks(np.arange(len(sorter_i)))
if SIM:
    ax.set_xticklabels(funa.SIM_head_ids[sorter_j], fontsize=5, rotation=90)
    ax.set_yticklabels(funa.SIM_head_ids[sorter_i], fontsize=5)
else:
    ax.set_xticklabels(funa.head_ids[sorter_j], fontsize=5, rotation=90)
    ax.set_yticklabels(funa.head_ids[sorter_i], fontsize=5)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
ax.set_xlim(-0.5, lim)

if stamp: pp.provstamp(ax,-.1,-.1," ".join(sys.argv))
if merge_bilateral:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/variability_merged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/variability_merged.pdf")
else:
    #plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/variability_unmerged.pdf")
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/variability_unmerged.pdf")
fig1.clf()



print("done")
