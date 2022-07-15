import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm

sort = "--sort" in sys.argv
neu_j = None
for s in sys.argv:
    sa = s.split(":")
    if sa[0] in ["-j","--j"]: neu_j = int(sa[1])

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="mutant",
                verbose=False)
                
occ1, occ2 = funa.get_occurrence_matrix()
occ3 = funa.get_observation_matrix()

time1 = np.linspace(0,30,1000)
time2 = np.linspace(0,200,1000)
dFF = funa.get_max_deltaFoverF(occ2,time1)
rise_times = funa.get_eff_rise_times(occ2,time2)
decay_times = funa.get_eff_decay_times(occ2,time2)
peak_times = funa.get_peak_times(occ2,time2)
avg_rise_times, rele_rise_times = funa.weighted_avg_occ2style(rise_times,dFF,return_rele=True)
avg_decay_times, rele_decay_times = funa.weighted_avg_occ2style(decay_times,dFF,return_rele=True)
avg_peak_times, rele_peak_times = funa.weighted_avg_occ2style(peak_times,dFF,return_rele=True)


if neu_j is None: neu_js = np.arange(funa.n_neurons)
else: neu_js = [neu_j]

for neu_j in neu_js:
    if len(avg_rise_times[:,neu_j])==0: continue
    if np.all(np.isnan(avg_rise_times[:,neu_j])): continue
    fig = plt.figure(neu_j+1)
    ax = fig.add_subplot(111)
    ax.hist(avg_rise_times[:,neu_j])
    ax.set_xlim(0,120)
    ax.set_title(funa.neuron_ids[neu_j])
plt.show()
