import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
import seaborn as sns

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
correct_decaying = "--correct-decaying" in sys.argv
if not correct_decaying: print("NOT USING THE NEW CORRECTION OF DECAYING RESPONSES")
two_min_occ = "--two-min-occ" in sys.argv
enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv

dst = None
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
    if sa[0] == "--dst": dst = sa[1]

print("nan_th",nan_th)
print("ds_exclude_tags",ds_exclude_tags)

# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True,
                 "smooth_mode": smooth_mode,
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,
                 "photobl_appl":True,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

signal_kwargs = {"remove_spikes": False,  "smooth": False,
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

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ, req_auto_response=req_auto_response)

occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=req_auto_response)
occ3_full = np.copy(occ3)

funa.load_aconnectome_from_file(chem_th=0, gap_th=0)
aconn_c = funa.aconn_chem
aconn_g = funa.aconn_gap

dys = []
distances = []
dy_at0 = []
dy_around0 = []
spike_traces = []
spike_traces_20_30 = []
synapses = []
gap_junctions = []


#for each pair of neurons for each stim event save the distance between the two and also the average df/f averaged across 30s after the stim
for ai in np.arange(funa.n_neurons):
    for aj in np.arange(funa.n_neurons):

        if two_min_occ:
            if occ1[ai, aj] < 2: continue

        for occ in occ2[ai, aj]:
            ds = occ["ds"]
            if ds in ds_exclude_i: continue
            ie = occ["stim"]
            i = occ["resp_neu_i"] #responding neuron
            j = funa.fconn[ds].stim_neurons[ie]  # stimulated neuron

            # Build the time axis
            i0 = funa.fconn[ds].i0s[ie]
            i1 = funa.fconn[ds].i1s[ie]
            shift_vol = funa.fconn[ds].shift_vol
            Dt = funa.fconn[ds].Dt

            y = funa.sig[ds].get_segment(i0, i1, baseline=False,
                                         normalize="none")[:, i]
            nan_mask = funa.sig[ds].get_segment_nan_mask(i0, i1)[:, i]


            vol_i = 0

            distance = funa.brains[ds].get_distance(vol_i, i, j)
            rec = wormdm.data.recording(funa.ds_list[ds])
            UmPerPixel = rec.frameUmPerPixel
            realdistance = distance*UmPerPixel
            if realdistance > 116:
                print("large")

            # if np.sum(nan_mask)>nan_th*len(y): continue
            # if not pp.Fconn.nan_ok(nan_mask,nan_th*len(y)): continue

            if signal_range is None:
                pre = np.average(y[:shift_vol-1])
                if pre == 0: continue

                if correct_decaying:
                    _, _, _, _, _, df_s_unnorm = funa.get_significance_features(
                        funa.sig[ds], i, i0, i1, shift_vol,
                        Dt, nan_th, return_traces=True)
                    if df_s_unnorm is None: continue
                    dy = np.average(df_s_unnorm) / pre
                    if np.isnan(dy):
                        continue
                    dys.append(dy)
                    distances.append(realdistance)
                    synapses.append(aconn_c[i,j])
                    gap_junctions.append(aconn_g[i,j])

                else:
                    dy = np.average(y[shift_vol:] - pre) / pre
                    dys.append(dy)
                    distances.append(realdistance)
            else:
                print("Not using corrected y")
                # std = np.std(y[:shift_vol-signal_range[0]])
                pre = np.average(y[:shift_vol])
                # dy = np.average(y[shift_vol-signal_range[0]:shift_vol+signal_range[1]+1] - pre)
                dy = np.average(np.abs(y[shift_vol - signal_range[0]:shift_vol + signal_range[1] + 1] - pre))
                dy /= pre
                dys.append(dy)
                distances.append(realdistance)
#sort all the arrays by the distance
distances_array = np.array(distances)
indicies = np.argsort(distances_array)
distances_sorted = distances_array[indicies]
distances_max = np.max(distances_sorted)
dys_sorted = np.array(dys)[indicies]
synapses_sorted = np.array(synapses)[indicies]
gap_junctions_sorted = np.array(gap_junctions)[indicies]
syn_plus_gj_sorted = synapses_sorted+gap_junctions_sorted

#determine the bins we want and then calculate the averages and std in each bin
#bins_edges = [1,10,20,30,40,50,60,70,80,90,100,110,120]
bins_edges = [1,4,10,16,22,28,34,40,46,52,58,64,70,76,82,88,94,100,106,112,118]
average_dys = []
stds_dys = []
indicies_edges = []
average_synapses = []
average_gap_junctions = []
average_syn_plus_gj = []
stds_synapses = []
stds_gap_junctions = []
stds_syn_plus_gj = []


for i in np.arange(len(bins_edges)):
    if i==0:
        indicies_in_bin = np.where(distances_sorted == 0)
        average_dys.append(np.mean(dys_sorted[indicies_in_bin]))
        stds_dys.append(np.std(dys_sorted[indicies_in_bin]))
        average_synapses.append(np.mean(synapses_sorted[indicies_in_bin]))
        average_gap_junctions.append(np.mean(gap_junctions_sorted[indicies_in_bin]))
        average_syn_plus_gj.append(np.mean(syn_plus_gj_sorted[indicies_in_bin]))
        stds_synapses.append(np.std(synapses_sorted[indicies_in_bin]))
        stds_gap_junctions.append(np.std(gap_junctions_sorted[indicies_in_bin]))
        stds_syn_plus_gj.append(np.std(syn_plus_gj_sorted[indicies_in_bin]))
        last_edge = max(indicies_in_bin[0])
        indicies_edges.append(last_edge)
        #last_edge_first = max(indicies_in_bin[0])

    elif i==1:
        indicies_in_bin = np.where(distances_sorted < bins_edges[i])
        last_edge = max(indicies_in_bin[0])
        indicies_edges.append(last_edge)
        last_edge_first = max(indicies_in_bin[0])
    else:
        indicies_in_bin = np.where(distances_sorted < bins_edges[i])
        indicies_in_bin = indicies_in_bin[0][last_edge+1:]
        average_dys.append(np.mean(dys_sorted[indicies_in_bin]))
        stds_dys.append(np.std(dys_sorted[indicies_in_bin]))
        average_synapses.append(np.mean(synapses_sorted[indicies_in_bin]))
        average_gap_junctions.append(np.mean(gap_junctions_sorted[indicies_in_bin]))
        average_syn_plus_gj.append(np.mean(syn_plus_gj_sorted[indicies_in_bin]))
        stds_synapses.append(np.std(synapses_sorted[indicies_in_bin]))
        stds_gap_junctions.append(np.std(gap_junctions_sorted[indicies_in_bin]))
        stds_syn_plus_gj.append(np.std(syn_plus_gj_sorted[indicies_in_bin]))
        last_edge = max(indicies_in_bin)
        indicies_edges.append(last_edge)



print("donewithaverages")
#the following is the same but if we had bins in which there is an equal number of values
def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))
plt.figure()
n_equal, bins_equal, patches_equal = plt.hist(distances[last_edge_first:], equalObs(distances[last_edge_first:], 10), edgecolor='black')
plt.clf()



average_dys_equal = []
stds_dys_equal = []





for i in np.arange(len(bins_equal)):
    if i==0:
        indicies_in_bin = np.where(distances_sorted < 1)
        average_dys_equal.append(np.mean(dys_sorted[indicies_in_bin]))
        stds_dys_equal.append(np.std(dys_sorted[indicies_in_bin]))
        last_edge = max(indicies_in_bin[0])

    else:
        indicies_in_bin = np.where(distances_sorted < bins_equal[i])
        indicies_in_bin = indicies_in_bin[0][last_edge+1:]
        average_dys_equal.append(np.mean(dys_sorted[indicies_in_bin]))
        stds_dys_equal.append(np.std(dys_sorted[indicies_in_bin]))
        last_edge = max(indicies_in_bin)


#setting the labels for plotting
#distances_labels = ['$0\mu m$', '$4 \mu m$-$10 \mu m$', '$10\mu m$-$16\mu m$','$16\mu m$-$22\mu m$','$22\mu m$-$28\mu m$', '$28\mu m$-$34\mu m$',
#                    '$34\mu m$-$40\mu m$', '$40\mu m$-$46\mu m$','$46\mu m$-$52\mu m$','$52\mu m$-$58\mu m$', '$58\mu m$-$64\mu m$','$64\mu m$-$70\mu m$',
#                    '$70\mu m$-$76\mu m$','$76\mu m$-$82\mu m$','$82\mu m$-$88\mu m$','$88\mu m$-$94\mu m$','$100\mu m$-$106\mu m$',
#                    '$106\mu m$-$112\mu m$','$112\mu m$-$118\mu m$']
distances_labels = ['0', '4-10', '10-16','16-22','22-28', '28-34', '34-40', '40-46','46-52','52-58', '58-64','64-70',
                    '70-76','76-82','82-88','88-94','94-100','100-106','106-112','112-118']

distances_labels_equal = ['$0\mu m$', '$0.01 \mu m$-$8.64 \mu m$', '$8.65\mu m$-$12.11\mu m$','$12.12\mu m$-$15.08\mu m$',"$15.09\mu m$-$18.18\mu m$",
                    '$18.19\mu m$-$21.68\mu m$','$21.69\mu m$-$25.63\mu m$', '$25.64\mu m$-$30.44\mu m$','$30.45\mu m$-$36.71\mu m$', '$36.71\mu m$-$46.41\mu m$',
                    '$46.42\mu m$-$117.80\mu m$']


x_pos = np.arange(len(distances_labels))
x_pos_equal = np.arange(len(distances_labels_equal))



spike_traces_sorted = np.array(spike_traces)[indicies]
indicies_in_bin1 = np.where(distances_sorted == 0)
spike_traces_bin1= spike_traces_sorted[0:max(indicies_in_bin1[0])]
spike_traces_bin1_average = np.mean(spike_traces_bin1, axis =0)
time_short = [-5. , -4.5, -4. , -3.5, -3. , -2.5, -2. , -1.5, -1. , -0.5,  0. ,
        0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5,  4. ,  4.5,  5. ]




dys_average_0 = dys_sorted[np.where(distances_sorted==0)]
dys_average_4_10 = dys_sorted[np.where(np.logical_and(4 <= distances_sorted, distances_sorted < 10))]
dys_average_10_16 = dys_sorted[np.where(np.logical_and(10 <= distances_sorted, distances_sorted < 16))]
#histogram of dys in different distance bins
fig10 = plt.figure()
plt.hist(dys_average_0, 40, histtype='step', label="$0\mu m$, n="+str(len(dys_average_0)), density = True, linewidth = 3)
plt.hist(dys_average_4_10, 40, histtype='step', label="$4 \mu m$-$10 \mu m$, n="+str(len(dys_average_4_10)), density = True, linewidth = 3)
plt.hist(dys_average_10_16, 40, histtype='step', label="$10\mu m$-$16\mu m$, n="+str(len(dys_average_10_16)), density = True, linewidth = 3)
plt.legend(fontsize = 15)
plt.ylabel('Density', fontsize = 20)
plt.xlabel('Average $\Delta F/F_0$', fontsize = 20)
plt.xlim(-0.5, 2.5)
plt.xticks([-0.5, 0.0,0.5,1.0,1.5,2.0,2.5], fontsize=15)
plt.yticks([0,2.5,5],fontsize=15)
plt.tight_layout()

#CDF cropped
fig11 = plt.figure()
plt.hist(dys_average_0, 40, histtype='step', label="$0\mu m$", density = True, cumulative= True, linewidth = 3)
plt.hist(dys_average_4_10, 40, histtype='step', label="$4 \mu m$-$10 \mu m$", density = True, cumulative= True, linewidth = 3)
plt.hist(dys_average_10_16, 40, histtype='step', label="$10\mu m$-$16\mu m$", density = True, cumulative= True, linewidth = 3)
plt.legend(fontsize =15, loc ="lower right")
plt.ylabel('CDF', fontsize =20)
plt.xlabel('Average $\Delta F/F_0$', fontsize =20)
plt.axvline(0.1, color = "black", linewidth = 1)
plt.xlim(-0.5, 2.5)
plt.xticks([-0.5, 0.0,0.5,1.0,1.5,2.0,2.5], fontsize=20)
plt.yticks([0,0.5,1],fontsize=20)
plt.tight_layout()
#CDF uncropped
fig12 = plt.figure()
plt.hist(dys_average_0, 40, histtype='step', label="$0\mu m$", density = True, cumulative= True, linewidth = 3)
plt.hist(dys_average_4_10, 40, histtype='step', label="$4 \mu m$-$10 \mu m$", density = True, cumulative= True, linewidth = 3)
plt.hist(dys_average_10_16, 40, histtype='step', label="$10\mu m$-$16\mu m$", density = True, cumulative= True, linewidth = 3)
plt.legend(fontsize =15, loc ="lower right")
plt.ylabel('CDF', fontsize =20)
plt.xlabel('Average $\Delta F/F_0$', fontsize =20)
plt.axvline(0.1, color = "black", linewidth = 1)
plt.xticks([-1, 0,1,2,3,4], fontsize=15)
plt.yticks([0,0.5,1],fontsize=15)
plt.tight_layout()

print("done")
