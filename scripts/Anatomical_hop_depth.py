import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "mutant"
ds_exclude_i = []
inclall_occ = "--inclall-occ" in sys.argv
signal_range = None
smooth_mode = "sg_causal"
smooth_n = 13
smooth_poly = 1
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
nan_th = 0.3
save = "--no-save" not in sys.argv

enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv
to_paper = "--to-paper" in sys.argv
plot = "--no-plot" not in sys.argv
figsize = (12, 10)

for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--nan-th": nan_th = float(sa[1])
    if sa[0] == "--ds-exclude-tags":
        ds_exclude_tags = sa[1]
        if ds_exclude_tags == "None": ds_exclude_tags = None
    if sa[0] == "--ds-tags": ds_tags = sa[1]
    if sa[0] == "--ds-exclude-i": ds_exclude_i = [int(sb) for sb in sa[1].split(",")]
    if sa[0] == "--signal-range":
        sb = sa[1].split(",")
        signal_range = [int(sbi) for sbi in sb]
    if sa[0] == "--figsize": figsize = [int(sb) for sb in sa[1].split(",")]

# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True, "smooth": True,
                 "smooth_mode": smooth_mode,
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file, "photobl_appl":True}



funa = pp.Funatlas.from_datasets(
    ds_list,
    merge_bilateral=merge_bilateral,
    merge_dorsoventral=False,
    merge_numbered=False, signal="green",
    signal_kwargs=signal_kwargs,
    ds_tags=ds_tags, ds_exclude_tags=ds_exclude_tags,
    enforce_stim_crosscheck=enforce_stim_crosscheck,
    verbose=False)

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ, req_auto_response=req_auto_response)
# If occ2 needs to be filtered
occ1, occ2 = funa.filter_occ12_from_sysargv(occ2, sys.argv)

occ3 = funa.get_observation_matrix(req_auto_response=req_auto_response)

if inclall_occ: inclall_occ2 = occ2
qvalues = funa.get_kolmogorov_smirnov_q(inclall_occ2)
qvalues_head = funa.reduce_to_head(qvalues)


funa.load_extrasynaptic_connectome_from_file()
esconn_ma = funa.esconn_ma # Monoamines
esconn_np = funa.esconn_np # Neuropeptides
esconn = np.logical_or(esconn_ma,esconn_np)

#esconn_ma = funatlas.reduce_to_head(esconn_ma)
#esconn_np = funatlas.reduce_to_head(esconn_np)
#esconn = funatlas.reduce_to_head(esconn)

#loading the connectome and the separate boolean connectomes for number of hops and reducing them all to head
funa.load_aconnectome_from_file(chem_th=0, gap_th=0)
aconn_c = funa.aconn_chem
aconn_g = funa.aconn_gap
one_hop_aconn = funa.get_n_hops_aconn(1)
one_hop_head = funa.reduce_to_head(one_hop_aconn)
two_hop_aconn = funa.get_n_hops_aconn(2)
two_hop_head = funa.reduce_to_head(two_hop_aconn)
three_hop_aconn = funa.get_n_hops_aconn(3)
three_hop_head = funa.reduce_to_head(three_hop_aconn)
four_hop_aconn = funa.get_n_hops_aconn(4)
four_hop_head = funa.reduce_to_head(four_hop_aconn)
sig = 0.05

# since diagonals do not actually count as "connections" I am replacing them by nans
qvalues_head_nodiag = np.copy(qvalues_head)
np.fill_diagonal(qvalues_head_nodiag,np.nan)

sig_ids = np.where(qvalues_head_nodiag < 0.05)

#calculating the n hop connectomes for significant connections
one_hop_sig = one_hop_head[sig_ids]
two_hop_sig = two_hop_head[sig_ids]
three_hop_sig = three_hop_head[sig_ids]
four_hop_sig = four_hop_head[sig_ids]

all_qs_ids = np.where(~np.isnan(qvalues_head_nodiag))
one_hop_justqs = one_hop_head[all_qs_ids]
two_hop_justqs = two_hop_head[all_qs_ids]
three_hop_justqs = three_hop_head[all_qs_ids]
four_hop_justqs = four_hop_head[all_qs_ids]

# calculating the probability of having a connection given n hops (the number of significant functional connections
# with n hops over the amount of all connections with a q value with n hops
prob_sig_given_one_hop = np.sum(one_hop_sig)/np.sum(one_hop_justqs)
prob_sig_given_two_hop = np.sum(two_hop_sig)/np.sum(two_hop_justqs)
prob_sig_given_three_hop = np.sum(three_hop_sig)/np.sum(three_hop_justqs)
prob_sig_given_four_hop = np.sum(four_hop_sig)/np.sum(four_hop_justqs)

fig1 = plt.figure(figsize=(4, 5))
ax = plt.gca()
probs = [prob_sig_given_one_hop,prob_sig_given_two_hop,prob_sig_given_three_hop,prob_sig_given_four_hop]
labels = ['1','2','3','4']
ax.bar(labels, probs)
plt.xticks(np.arange(0, 4, step=1), fontsize= 25)
ax.set_xlabel('Min Anatomical Path Length',fontsize= 25)
ax.set_ylabel('Probability of Functional Connection \n Given n Hops',fontsize= 25)
plt.yticks(np.arange(0, 0.181, step=0.09), fontsize= 25)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/AconnHopDepth2.pdf", bbox_inches='tight')
fig1.clf()

iids = jids = funa.neuron_ids

#min_anatomical_hops = np.zeros((funa.n_neurons, funa.n_neurons)) * np.nan
#for ai in np.arange(funa.n_neurons):
  #  for aj in np.arange(funa.n_neurons):
  #      if qvalues[ai,aj] < 0.05 and ai != aj:
  #          if aconn_c[ai,aj] > 0 or aconn_g[ai,aj] > 0:
  #              min_anatomical_hops[ai,aj] = 1

  #          else:
  #              potential_hops_c = np.where(aconn_c[:,aj]>0)
  #              potential_hops_g = np.where(aconn_g[:, aj]>0)
  #              possible_2hop_con = 0
  #              for ak in np.unique(np.concatenate((potential_hops_g[0],potential_hops_c[0]))):
  #                  if aconn_c[ai,ak] > 0 or aconn_g[ai,ak] > 0:
  #                      possible_2hop_con += 1
  #                      min_anatomical_hops[ai, aj] = 2
  #                      break
  #              if possible_2hop_con > 0:
  #                  min_anatomical_hops[ai, aj] = 2

  #              else:
  #                  possible_3hop_con = 0
  #                  for ak in np.unique(np.concatenate((potential_hops_g[0], potential_hops_c[0]))):
  #                      potential_hops_from_ak_c = np.where(aconn_c[:, ak] > 0)
  #                      potential_hops_from_ak_g = np.where(aconn_g[:, ak] > 0)
  #                      for al in np.unique(np.concatenate((potential_hops_from_ak_g[0], potential_hops_from_ak_c[0]))):
   #                         if aconn_c[ai, al] > 0 or aconn_g[ai, al] > 0:
   #                             possible_3hop_con += 1
  #                              min_anatomical_hops[ai, aj] = 3
  #                              break

  #                  if possible_3hop_con > 0:
  #                      min_anatomical_hops[ai, aj] = 3

  #                  else:
  #                      possible_4hop_con = 0
  #                      for ak in np.unique(np.concatenate((potential_hops_g[0], potential_hops_c[0]))):
  #                          potential_hops_from_ak_c = np.where(aconn_c[:, ak] > 0)
  #                          potential_hops_from_ak_g = np.where(aconn_g[:, ak] > 0)
  #                          for al in np.unique(np.concatenate((potential_hops_from_ak_g[0], potential_hops_from_ak_c[0]))):
  #                              potential_hops_from_al_c = np.where(aconn_c[:, al] > 0)
  #                              potential_hops_from_al_g = np.where(aconn_g[:, al] > 0)
  #                              for ah in np.unique(
  #                                      np.concatenate((potential_hops_from_al_g[0], potential_hops_from_al_c[0]))):
  #                                  if aconn_c[ai, ah] > 0 or aconn_g[ai, ah] > 0:
  #                                      possible_4hop_con += 1
 #                                       min_anatomical_hops[ai, aj] = 4
 #                                       break

# calculating the minimum anatomical hop matrix for the significant connections
min_anatomical_hops = np.zeros((funa.n_neurons, funa.n_neurons)) * np.nan
for ai in np.arange(funa.n_neurons):
    for aj in np.arange(funa.n_neurons):
        if qvalues[ai, aj] < 0.05 and ai != aj:
            if one_hop_aconn[ai, aj]:
               min_anatomical_hops[ai, aj] = 1

            elif two_hop_aconn[ai, aj]:
                min_anatomical_hops[ai, aj] = 2

            elif three_hop_aconn[ai, aj]:
                min_anatomical_hops[ai, aj] = 3

            elif four_hop_aconn[ai, aj]:
                min_anatomical_hops[ai, aj] = 4


# calculating the min anatomical matrix for all connections with a q value
min_anatomical_hops_all = np.zeros((funa.n_neurons, funa.n_neurons)) * np.nan
for ai in np.arange(funa.n_neurons):
    for aj in np.arange(funa.n_neurons):
        if ai != aj and ~np.isnan(qvalues[ai, aj]):
            if one_hop_aconn[ai, aj]:
               min_anatomical_hops_all[ai, aj] = 1

            elif two_hop_aconn[ai, aj]:
                min_anatomical_hops_all[ai, aj] = 2

            elif three_hop_aconn[ai, aj]:
                min_anatomical_hops_all[ai, aj] = 3

            elif four_hop_aconn[ai, aj]:
                min_anatomical_hops_all[ai, aj] = 4

min_anatomical_hops_head = funa.reduce_to_head(min_anatomical_hops)
min_anatomical_hops_flat = min_anatomical_hops_head.flatten("C")
min_anatomical_hops_all_head = funa.reduce_to_head(min_anatomical_hops_all)
min_anatomical_hops_all_flat = min_anatomical_hops_all_head.flatten("C")

meanhops_sig = np.mean(min_anatomical_hops_flat[~np.isnan(min_anatomical_hops_flat)])
meanhops_all = np.mean(min_anatomical_hops_all_flat[~np.isnan(min_anatomical_hops_all_flat)])

fig5 = plt.figure(figsize=(4, 5))
ax = plt.gca()
n_sig, bins, patches = ax.hist(min_anatomical_hops_flat, bins = [0.5,1.5,2.5,3.5,4.5], density=True, label='q < 0.05 connections', rwidth = 1, alpha = 1)
#n_all, bins, patches = ax.hist(min_anatomical_hops_all_flat, bins = [0.5,1.5,2.5,3.5,4.5], density=True, label='All Possible Pairs', rwidth = 0.7, alpha = 0.5)
#n_sig, bins, patches = ax.hist(min_anatomical_hops_flat, bins = [0.5,1.5,2.5,3.5,4.5], density=True, histtype='step', label='Pairs with q < 0.05', linewidth = 5)
n_all, bins, patches = ax.hist(min_anatomical_hops_all_flat, bins = [0.5,1.5,2.5,3.5,4.5], density=True, histtype='step', label='all possible pairs', linewidth = 5)
plt.xticks(np.arange(1, 5, step=1), fontsize= 25)
ax.set_xlabel('Min Anatomical Path Length',fontsize= 25)
ax.set_ylabel('Density of Pairs',fontsize= 25)
plt.yticks(np.arange(0, 0.51, step=0.25), fontsize= 25)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
labels = ['q < 0.05 connections','all possible pairs']
plt.legend(labels, fontsize = 15)
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/AconnHopDepth.pdf", bbox_inches='tight')
fig5.clf()
print("done")

