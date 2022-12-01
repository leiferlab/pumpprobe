import matplotlib.pyplot as plt
import numpy as np
import pumpprobe as pp
import sys
import pandas as pd
import matplotlib.cm as cm
import wormdatamodel as wormdm
import wormbrain as wormb
from scipy import stats
import scipy as sp
import matplotlib.pylab as pl
from pumpprobe import provenance as prov

# List of datasets
fname = "/projects/LEIFER/Sophie/AFDAIYtestList.txt"

foldername = ["/projects/LEIFER/Sophie/NewRecordings/20211117/pumpprobe_20211117_150032", "/projects/LEIFER/Sophie/NewRecordings/20211119/pumpprobe_20211119_142937"]

signal = "green"

signal_kwargs = {"remove_spikes": True,  "smooth": True, "smooth_mode": "sg",
                 "smooth_n": 13, "smooth_poly": 1, "matchless_nan_th": 0.3, }
signal_kwargs2 = {"remove_spikes": True,  "smooth": True, "smooth_mode": "sg",
                 "smooth_n": 13, "smooth_poly": 1, "matchless_nan_th": 0}
# Build Funatlas object
funatlas = pp.Funatlas.from_datasets(
                fname,merge_bilateral=False,merge_dorsoventral=False,
                merge_numbered=False,signal=signal,signal_kwargs=signal_kwargs)
funatlas2 = pp.Funatlas.from_datasets(
                fname,merge_bilateral=False,merge_dorsoventral=False,
                merge_numbered=False,signal=signal,signal_kwargs=signal_kwargs2)

nr_of_datasets = len(funatlas.ds_list)
actual_AIYS = {}

#actual_AIYS["0"] = ["AIY1","AIY3", "AIY5"]
actual_AIYS["0"] = ["AIY1", "AIY5"]
stimulated = []
actual_AIYS["1"] = ["AIY4", "AIY3"]

datasets_where_AFD_response_is_nice = [0,1]

AIY_Peaks_responding = []
AIY_Peaks_fit_responding = []
AFD_Peaks_responding = []
AFD_Peaks_fit_responding = []
Stim_number = []
N_pulses =[]

for ds in np.arange(nr_of_datasets):
    stims = funatlas.fconn[ds].stim_neurons #this is a list of all the stimulations, the number is the neuron that was stimulated, the index is the stimulation number
    rec = wormdm.data.recording(foldername[ds])
    events = rec.get_events()
    cervelli = wormb.Brains.from_file(foldername[ds], ref_only=True, verbose=False)
    labels = cervelli.get_labels(0)
    npulses = events["optogenetics"]["properties"]["n_pulses"]
    AIYS_name = []
    AIYS_index = []
    for k in np.arange(len(labels)):
        if labels[k] in actual_AIYS[str(ds)]:
            AIYS_name.append(labels[k])
            AIYS_index.append(k)

    fig4 = plt.figure(1, figsize=(30, 15))
    ax1 = fig4.add_subplot(121)
    ax2 = fig4.add_subplot(122)


    for ie in np.arange(len(stims)):
        i0 = funatlas.fconn[ds].i0s[ie]
        i1 = funatlas.fconn[ds].i1s[ie]
        Dt = funatlas.fconn[ds].Dt
        stim_j = funatlas.fconn[ds].stim_neurons[ie]
        responding = funatlas.fconn[ds].resp_neurons_by_stim[ie]
        shift_vol = funatlas.fconn[ds].shift_vol
        time_trace = (np.arange(i1 - i0) - shift_vol) * Dt
        y_trace_stim = funatlas2.sig[ds].get_segment(i0, i1, shift_vol, normalize="", baseline_mode="exp")[:,
                       stim_j]
        peak_stim = max(y_trace_stim, key=abs)
        time = (np.arange(62)) * Dt  # this is the time axis starting from 0, from the time_0_vol
        #if not np.array_equal(funatlas2.fconn[ds].fit_params[ie][stim_j]['params'],
        #                     funatlas2.fconn[ds].fit_params_default['params']) or not \
        #funatlas2.fconn[ds].fit_params[ie][stim_j]['n_branches'] == funatlas2.fconn[ds].fit_params_default[
        #    'n_branches'] or not np.array_equal(funatlas2.fconn[ds].fit_params[ie][stim_j]['n_branch_params'],
        #                                       funatlas2.fconn[ds].fit_params_default['n_branch_params']):
            # Generate the fitted stimulus activity
        stim_unc_par = funatlas.fconn[ds].fit_params_unc[ie][stim_j]  # what is the second index
        if stim_unc_par["n_branches"] == 0:
            print("skipping " + str(ie))
            continue

        else:

            # y = funatlas.fconn[ds].get_kernel(time, ie, resp_i)
            stim_ec = pp.ExponentialConvolution.from_dict(stim_unc_par)
            fitted_stim = stim_ec.eval(time)
            peak_stim_fit = max(fitted_stim, key=abs)

            colors = pl.cm.Blues(np.linspace(0, 1, len(stims)))
            ax1.plot(time, fitted_stim, color=colors[int(((npulses[ie] / 500000) * 20))],
                     label="duration of pulse train: " + str(npulses[ie] / 500000))
            ax2.plot(time, fitted_stim, color=colors[ie],
                     label="duration of pulse train: " + str(npulses[ie] / 500000))





        for a in np.arange(len(AIYS_name)):
            y_trace_resp = funatlas.sig[ds].get_segment(i0, i1, shift_vol, normalize="", baseline_mode="exp")[:, AIYS_index[a]]
            peak_resp = max(y_trace_resp, key=abs)
            if AIYS_index[a] in responding:
                #if not np.array_equal(funatlas2.fconn[ds].fit_params[ie][AIYS_index[a]]['params'],
                #                      funatlas2.fconn[ds].fit_params_default['params']) or not \
                #        funatlas2.fconn[ds].fit_params[ie][AIYS_index[a]]['n_branches'] == funatlas2.fconn[ds].fit_params_default[
                #            'n_branches'] or not np.array_equal(funatlas2.fconn[ds].fit_params[ie][AIYS_index[a]]['n_branch_params'], funatlas2.fconn[ds].fit_params_default['n_branch_params']):

                    # Generate the fitted stimulus activity
                resp_unc_par = funatlas.fconn[ds].fit_params_unc[ie][AIYS_index[a]]  # what is the second index
                if resp_unc_par["n_branches"] == 0:
                    print("skipping " + str(ie))
                    continue

                else:

                    # y = funatlas.fconn[ds].get_kernel(time, ie, resp_i)
                    resp_ec = pp.ExponentialConvolution.from_dict(resp_unc_par)
                    fitted_resp = resp_ec.eval(time)
                    peak_resp_fit = max(fitted_resp, key=abs)

                    AIY_Peaks_responding.append(peak_resp)
                    AIY_Peaks_fit_responding.append(peak_resp_fit)
                    AFD_Peaks_responding.append(peak_stim)
                    AFD_Peaks_fit_responding.append(peak_stim_fit)
                    N_pulses.append(npulses[ie])
                    Stim_number.append(ie)
    ax1.set_xlabel("Time (s)", fontsize=20)
    ax2.set_xlabel("Time (s)", fontsize=20)
    ax1.set_ylabel("Calcium Signal (arb. u.)", fontsize=20)
    ax2.set_ylabel("Calcium Signal (arb. u.)", fontsize=20)
    ax1.set_title("Colored by Pulse Duration",  fontsize=20)
    ax2.set_title("Colored by Stimulus Number", fontsize=20)
    ax1.legend(fontsize=15)
    ax2.legend(fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    prov.stamp(ax1, -.1, -.1, __file__)
    fig4.suptitle("AFD Calcium Auto-Response to Stimulation of Different Pulse Train Durations", fontsize=30)
    fig4.savefig("/projects/LEIFER/francesco/funatlas/figures/AFD/AFD_Calc" + str(ds) + ".png",
                 bbox_inches="tight")
    fig4.clf()
npulses_to_use = np.sort(npulses)

AFD_peaks_ave = []
AFD_peaks_std = []
AIY_peaks_ave = []
AIY_peaks_std = []
AFD_peaks_ave_fit = []
AFD_peaks_std_fit = []
AIY_peaks_ave_fit = []
AIY_peaks_std_fit = []
n_pulses_to_use_fig = []
for n in npulses_to_use:
    AFD_n = []
    AIY_n = []
    AFD_n_fit = []
    AIY_n_fit = []
    for m in np.arange(len(AFD_Peaks_responding)):
        if N_pulses[m] == n:
            AFD_n.append(AFD_Peaks_responding[m])
            AIY_n.append(AIY_Peaks_responding[m])
            AFD_n_fit.append(AFD_Peaks_fit_responding[m])
            AIY_n_fit.append(AIY_Peaks_fit_responding[m])
    if AFD_n != [] and AIY_n != [] and AFD_n_fit != [] and AIY_n_fit != []:
        AFD_peaks_ave.append(np.mean(AFD_n))
        AFD_peaks_std.append(np.std(AFD_n))
        AIY_peaks_ave.append(np.mean(AIY_n))
        AIY_peaks_std.append(np.std(AIY_n))
        AFD_peaks_ave_fit.append(np.mean(AFD_n_fit))
        AFD_peaks_std_fit.append(np.std(AFD_n_fit))
        AIY_peaks_ave_fit.append(np.mean(AIY_n_fit))
        AIY_peaks_std_fit.append(np.std(AIY_n_fit))
        n_pulses_to_use_fig.append(n)


from scipy.optimize import curve_fit
def fitf(x,a):
    return a*x
    
a,_ = curve_fit(fitf,AFD_peaks_ave,AIY_peaks_ave,p0=[1],sigma=AIY_peaks_std)
fitline_AFD_comp = fitf(AFD_peaks_ave,a)

#fitline_model_AFD_comp = np.polyfit(AFD_peaks_ave, AIY_peaks_ave, 1, w=AIY_peaks_std)
#fitline_AFD_comp = [(element * fitline_model_AFD_comp[0]) + fitline_model_AFD_comp[1] for element in AFD_peaks_ave]

# figure of ave AIY peak vs ave AFD peak responding only with errorbars
fig = plt.figure(1, figsize=(15, 15))
ax = fig.add_subplot()
#plt.scatter(AFD_Peaks, AIY_Peaks, alpha=0.5)
ax.plot(AFD_peaks_ave, fitline_AFD_comp, linewidth = 6)
ax.errorbar(AFD_peaks_ave, AIY_peaks_ave, xerr = AFD_peaks_std, yerr = AIY_peaks_std, fmt='o', color = "blue", ecolor = "lightgray", elinewidth = 6, markersize=20)
ax.set_xlabel("Change in AFD Signal", fontsize = 40)
ax.set_ylabel("Change in AIY Signal ", fontsize = 40)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
prov.stamp(ax,-.1,-.1,__file__)
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/AFD/AFD_vs_AIY_errorbars.png",
                bbox_inches="tight")

fig.clf()


a,_ = curve_fit(fitf,AFD_peaks_ave_fit,AIY_peaks_ave_fit,[1])
fitline_AFD_comp_fit = fitf(AFD_peaks_ave_fit,a)

#fitline_model_AFD_comp_fit = np.polyfit(AFD_peaks_ave_fit, AIY_peaks_ave_fit, 1, w=AIY_peaks_std_fit)
#fitline_AFD_comp_fit = [(element * fitline_model_AFD_comp_fit[0]) + fitline_model_AFD_comp_fit[1] for element in AFD_peaks_ave_fit]

# figure of ave AIY peak vs ave AFD peak responding only with errorbars of the fit peaks
fig2 = plt.figure(2, figsize=(5,5))
plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)
ax = fig2.add_subplot(111)
#plt.scatter(AFD_Peaks, AIY_Peaks, alpha=0.5)
ax.plot(np.append(0,AFD_peaks_ave_fit), np.append(0,fitline_AFD_comp_fit))
ax.errorbar(AFD_peaks_ave_fit, AIY_peaks_ave_fit, xerr = AFD_peaks_std_fit, yerr = AIY_peaks_std_fit, fmt='o', color = "blue", ecolor = "lightgray")
ax.set_xlabel("$\Delta$F AFD", fontsize = 40)
ax.set_ylabel("$\Delta$F AIY", fontsize = 40)
ax.set_xlim(0,)
ax.set_ylim(0,)
ax.set_xticks([0,1,2,3,4])
#ax.set_xticks(fontsize = 30)
#ax.set_yticks(fontsize = 30)
#prov.stamp(ax,-.1,-.1,__file__)
fig2.tight_layout()
fig2.savefig("/projects/LEIFER/francesco/funatlas/figures/AFD/AFD_vs_AIY_errorbars_fit.png",
                bbox_inches="tight")
fig2.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS11/AFD_vs_AIY_errorbars_fit.pdf", dpi=300,
                bbox_inches="tight")

fig2.clf()

fitline_model_AFD_comp_fit_scat = np.polyfit(AFD_Peaks_fit_responding, AIY_Peaks_fit_responding, 1)
x_axis = np.arange(5)
fitline_AFD_comp_fit_scat = [(element * fitline_model_AFD_comp_fit_scat[0]) + fitline_model_AFD_comp_fit_scat[1] for element in x_axis]

# no error just a scatterplot
fig3 = plt.figure(1, figsize=(15, 15))
ax = fig3.add_subplot()
plt.scatter(AFD_Peaks_fit_responding, AIY_Peaks_fit_responding, alpha=0.5, s=200)
plt.plot(x_axis, fitline_AFD_comp_fit_scat, linewidth = 6)
#plt.errorbar(AFD_peaks_ave_fit, AIY_peaks_ave_fit, xerr = AFD_peaks_std_fit, yerr = AIY_peaks_std_fit, fmt='o', color = "blue", ecolor = "lightgray", elinewidth = 2)
ax.set_xlabel("Change in AFD Signal", fontsize = 40)
ax.set_ylabel("Change in AIY Signal", fontsize = 40)
ax.set_xlim(0,)
plt.xticks(fontsize = 40)
plt.yticks(fontsize = 40)
prov.stamp(ax,-.1,-.1,__file__)
fig3.savefig("/projects/LEIFER/francesco/funatlas/figures/AFD/AFD_vs_AIY_scatter_fit.png",
                bbox_inches="tight")
fig3.clf()

npulses_to_use_in_s = []
for k in n_pulses_to_use_fig:
    npulses_to_use_in_s.append(k/500000)

fitline_model_AIY_npulses_comp_fit = np.polyfit(npulses_to_use_in_s, AIY_peaks_ave_fit, 1, w=AIY_peaks_std_fit)
fitline_AIY_npulses_comp_fit = [(element * fitline_model_AIY_npulses_comp_fit[0]) + fitline_model_AIY_npulses_comp_fit[1] for element in npulses_to_use_in_s]

#  errorplot of npulses vs AIY
fig4 = plt.figure(1, figsize=(15, 15))
ax = fig4.add_subplot()
#plt.scatter(AFD_Peaks_fit_responding, AIY_Peaks_fit_responding, alpha=0.5)
plt.plot(npulses_to_use_in_s, fitline_AIY_npulses_comp_fit, linewidth = 6)
plt.errorbar(npulses_to_use_in_s, AIY_peaks_ave_fit, yerr = AIY_peaks_std_fit, fmt='o', color = "blue", ecolor = "lightgray", elinewidth = 6, markersize=20)
ax.set_xlabel("Pulse Duration (s)", fontsize = 40)
ax.set_ylabel("Change in AIY Signal", fontsize = 40)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
prov.stamp(ax,-.1,-.1,__file__)
fig4.savefig("/projects/LEIFER/francesco/funatlas/figures/AFD/AIY_vs_npulses_fit.png",
                bbox_inches="tight")

fig4.clf()

fitline_model_AFD_npulses_comp_fit = np.polyfit(npulses_to_use_in_s, AFD_peaks_ave_fit, 1, w=AFD_peaks_std_fit)
fitline_AFD_npulses_comp_fit = [(element * fitline_model_AFD_npulses_comp_fit[0]) + fitline_model_AFD_npulses_comp_fit[1] for element in npulses_to_use_in_s]

#  errorplot of npulses vs AFD
fig5 = plt.figure(1, figsize=(15, 15))
ax = fig5.add_subplot()
#plt.scatter(AFD_Peaks_fit_responding, AIY_Peaks_fit_responding, alpha=0.5)
plt.plot(npulses_to_use_in_s, fitline_AFD_npulses_comp_fit, linewidth = 6)
plt.errorbar(npulses_to_use_in_s, AFD_peaks_ave_fit, yerr = AFD_peaks_std_fit, fmt='o', color = "blue", ecolor = "lightgray", elinewidth = 6, markersize=20)
ax.set_xlabel("Pulse Duration (s)", fontsize = 40)
ax.set_ylabel("Change in AFD Signal", fontsize = 40)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
prov.stamp(ax,-.1,-.1,__file__)
fig5.savefig("/projects/LEIFER/francesco/funatlas/figures/AFD/AFD_vs_npulses_fit.png",
                bbox_inches="tight")

fig5.clf()

#  errorplot of npulses vs AFD
fig6 = plt.figure(1, figsize=(15, 15))
ax = fig6.add_subplot()
sc = plt.scatter(AFD_Peaks_fit_responding, N_pulses, alpha=0.5, s=400, c=Stim_number, cmap="Blues")
plt.colorbar(sc)
#plt.plot(npulses_to_use_in_s, fitline_AFD_npulses_comp_fit)
#plt.errorbar(npulses_to_use_in_s, AFD_peaks_ave_fit, yerr = AFD_peaks_std_fit, fmt='o', color = "blue", ecolor = "lightgray", elinewidth = 2)
ax.set_xlabel("Pulse Duration (s)", fontsize = 40)
ax.set_ylabel("Change in AFD Signal", fontsize = 40)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
prov.stamp(ax,-.1,-.1,__file__)
fig6.savefig("/projects/LEIFER/francesco/funatlas/figures/AFD/AFD_vs_npulses_scatter.png",
                bbox_inches="tight")

fig6.clf()


print("hello")
