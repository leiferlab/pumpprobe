import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
from multipy import fdr


ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "mutant"
inclall_occ = "--inclall-occ" in sys.argv
signal_range = None
smooth_mode = "sg_causal"
smooth_n = 13
smooth_poly = 1
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
nan_th = 0.3
save = "--no-save" not in sys.argv
two_min_occ = "--two-min-occ" in sys.argv

enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv

# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,                 
                 "matchless_nan_th_from_file": matchless_nan_th_from_file, "photobl_appl":True}

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

occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=req_auto_response)

inclall_occ2 = occ2
#pvalues,_,_ = funa.get_kolmogorov_smirnov_p(inclall_occ2)
#pvalues_head = funa.reduce_to_head(pvalues)

#Convert pvalues to qvalues

#significant, q = fdr.qvalue(pvalues[np.isfinite(pvalues)]) #exclude the nans
#q_mat = np.copy(pvalues) #get the qvalues back into a matrix form
#q_mat[np.isfinite(pvalues)] = q
q_mat = funa.get_kolmogorov_smirnov_q(inclall_occ2)
q = q_mat[~np.isnan(q_mat)]
qvalues_head = funa.reduce_to_head(q_mat)
q_head = qvalues_head[~np.isnan(qvalues_head)]

# define the threshold for significance and calculate the total number of significant connections
significance_number = 0.05
Total_sig =(q_mat < significance_number).sum()
Total_sig_head =(qvalues_head < significance_number).sum()

iids = jids = funa.neuron_ids  # gives ids for each global index
headneurons = funa.head_ids
#first we want to look at the probability that two left right pairs are connected
LR_pairs_connected = 0
LR_pairs_notconnected = 0
LR_pairs_connected_head = 0
LR_pairs_notconnected_head = 0
Connected_both_LR = 0
Connected_not_both = 0
# for each neuron if it is L/R we look at if it is significantly connected to its R/L partner
pairs_connected = []
for iid in iids:
    for jid in jids:
        # Convert the requested IDs to atlas-indices.
        i, j = funa.ids_to_i([iid, jid])

        qval = q_mat[i,j]
        if iid.endswith("L"):
            if jid == iid[0:-1] + "R" and qval < significance_number:
                LR_pairs_connected += 1
                if iid in headneurons:
                    LR_pairs_connected_head += 1
                pairs_connected.append(iid[0:-1])
                print(iid)
            elif jid == iid[0:-1] + "R" and qval > significance_number:
                LR_pairs_notconnected += 1
                if iid in headneurons:
                    LR_pairs_notconnected_head += 1
        elif iid.endswith("R"):
            if jid == iid[0:-1] + "L" and qval < significance_number:
                LR_pairs_connected += 1
                if iid in headneurons:
                    LR_pairs_connected_head += 1
                pairs_connected.append(iid[0:-1])
                print(iid)
            elif jid == iid[0:-1] + "L" and qval > significance_number:
                LR_pairs_notconnected += 1
                if iid in headneurons:
                    LR_pairs_notconnected_head += 1

# here I am not double counting since a connection between i and j does not
# mean there has to be a connection between j and i
total_LR_pairs = LR_pairs_connected + LR_pairs_notconnected
P_LR_given_connected = LR_pairs_connected/Total_sig
P_connected_given_LR = LR_pairs_connected/total_LR_pairs
P_connected_total = Total_sig/len(q)

# next we check if one upstream neuron is connected to both downstream bilateral partners
Connected_both_LR = 0
Connected_not_both = 0
for iid in iids:
    for jid in jids:
        # Convert the requested IDs to atlas-indices.
        i, j = funa.ids_to_i([iid, jid])

        qval = q_mat[i, j]
        if iid.endswith("L") and qval < significance_number and iid != jid:
            # if the responding neuron is a left neuron we wwant to look at if there is also a significant connection to the right one
            otheriid = iid[0:-1] + "R" # the name of the right neuron
            k = funa.ids_to_i(otheriid) # the id number of the right neuron
            otherqval = q_mat[k, j] # qvalue of the connection to the right neuron
            if otherqval < significance_number:
                Connected_both_LR += 1
            #elif otherqval > significance_number or np.isnan(otherqval): #do we want to include the ones with no qval in the denominator?
            elif otherqval > significance_number:
                Connected_not_both +=1

        elif iid.endswith("R") and qval < significance_number and iid != jid:
            # if the responding neuron is a right neuron we wwant to look at if there is also a significant connection to the left one
            otheriid = iid[0:-1] + "L"  # the name of the left neuron
            k = funa.ids_to_i(otheriid)  # the id number of the left neuron
            otherqval = q_mat[k, j]  # qvalue of the connection to the left neuron
            if otherqval < significance_number:
                Connected_both_LR += 1
            # elif otherqval > significance_number or np.isnan(otherqval): #do we want to include the ones with no qval in the denominator?
            elif otherqval > significance_number:
                Connected_not_both += 1

total_could_connect_LR = (Connected_both_LR/2) + Connected_not_both
Prob_connected_both_LR = (Connected_both_LR/2)/total_could_connect_LR
# divided by two since I am double counting

# now we check if an upstream neuron is connected to a downstream neuron if its bilateral partner is also connected to that downstream neuron
Both_LR_connected = 0
Both_LR_not_connected = 0
for iid in iids:
    for jid in jids:
        # Convert the requested IDs to atlas-indices.
        i, j = funa.ids_to_i([iid, jid])

        qval = q_mat[i, j]
        if jid.endswith("L") and qval < significance_number and iid != jid:
            # if the stimulated neuron is a left neuron we want to look at if there is also a significant connection to the right stimulated one
            otherjid = jid[0:-1] + "R" # the name of the right stimulated neuron
            l = funa.ids_to_i(otherjid) # the id number of the right stimulated neuron
            otherqval = q_mat[i, l] # qvalue of the connection to the right stimulated neuron
            if otherqval < significance_number and otherjid != iid:
                Both_LR_connected += 1
            #elif otherqval > significance_number or np.isnan(otherqval): #do we want to include the ones with no qval in the denominator?
            elif otherqval > significance_number and otherjid != iid:
                Both_LR_not_connected +=1

        elif jid.endswith("R") and qval < significance_number and iid != jid:
            # if the stimulated neuron is a right neuron we want to look at if there is also a significant connection to the left stimulated one
            otherjid = jid[0:-1] + "L"  # the name of the left stimulated neuron
            l = funa.ids_to_i(otherjid)  # the id number of the left stimulated neuron
            otherqval = q_mat[i, l]  # qvalue of the connection to the left stimulated neuron
            if otherqval < significance_number and otherjid != iid:
                Both_LR_connected += 1
            # elif otherqval > significance_number or np.isnan(otherqval): #do we want to include the ones with no qval in the denominator?
            elif otherqval > significance_number and otherjid != iid:
                Both_LR_not_connected += 1


total_could_be_connected_to_both_upstream_LR = (Both_LR_connected/2) + Both_LR_not_connected
Prob_connected_both_upstream_LR = (Both_LR_connected/2)/total_could_be_connected_to_both_upstream_LR

fig = plt.figure(figsize=(10, 5))
#ax = fig.add_axes([0,0,1,1])
probs = ['P(Connection|Bilateral Pairs)', 'P(Connection)', 'P(Connection to Both'+"\n"+'Downstream Bilateral Neurons)', 'P(Connection to Both \n Upstream Bilateral Neurons)']
Probs_numbers = [P_connected_given_LR, P_connected_total, Prob_connected_both_LR, Prob_connected_both_upstream_LR]
y_pos = range(len(probs))
ypos = [2*i for i in y_pos]
plt.ylabel("Probability")
plt.bar(ypos, Probs_numbers)
plt.xticks(ypos, probs, rotation=0, fontsize = 10)

#plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/Miscellaneous_Supplement/LR_statistics.pdf")
plt.savefig("/projects/LEIFER/Sophie/Figures/Response_Statistics/LR_statistics.pdf")
fig.clf()

fig2 = plt.figure(figsize=(4, 4))
#ax = fig.add_axes([0,0,1,1])
#probs = ['P(Connection)', 'P(Connection|Bilateral Pairs)']
probs = ['All Pairs', 'Bilateral Pairs']
Probs_numbers = [P_connected_total, P_connected_given_LR]
y_pos = range(len(probs))
ypos = [i for i in y_pos]
ax = plt.gca()
plt.ylabel("Probability of q<0.05", fontsize = 25)
plt.bar(ypos, Probs_numbers)
plt.xticks(ypos, probs, rotation=0, fontsize = 25)
plt.yticks(np.arange(0, 0.51, step=0.25), fontsize= 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/LR_statistics.pdf", bbox_inches='tight')
plt.savefig("/projects/LEIFER/Sophie/Figures/Response_Statistics/LR_statistics_justtwo.pdf", bbox_inches='tight')
fig2.clf()
# after this point I am remaking the mappa of average DF to find out how many of our significant
# connections are inhibitory
mappa = np.zeros((funa.n_neurons, funa.n_neurons)) * np.nan
count = np.zeros((funa.n_neurons, funa.n_neurons))

for ai in np.arange(funa.n_neurons):
    for aj in np.arange(funa.n_neurons):
        if np.isnan(mappa[ai, aj]) and occ3[ai, aj] > 0:
            mappa[ai, aj] = 0.0

        if two_min_occ:
            if occ1[ai, aj] < 2: continue

        ys = []
        times = []
        confidences = []
        for occ in occ2[ai, aj]:
            ds = occ["ds"]
            ie = occ["stim"]
            i = occ["resp_neu_i"]

            # Build the time axis
            i0 = funa.fconn[ds].i0s[ie]
            i1 = funa.fconn[ds].i1s[ie]
            shift_vol = funa.fconn[ds].shift_vol

            y = funa.sig[ds].get_segment(i0, i1, baseline=False,
                                         normalize="none")[:, i]
            nan_mask = funa.sig[ds].get_segment_nan_mask(i0, i1)[:, i]

            if np.sum(nan_mask) > nan_th * len(y): continue

            if signal_range is None:
                pre = np.average(y[:shift_vol])
                if pre == 0: continue
                dy = np.average(y[shift_vol:] - pre) / pre
            else:
                # std = np.std(y[:shift_vol-signal_range[0]])
                pre = np.average(y[:shift_vol])
                # dy = np.average(y[shift_vol-signal_range[0]:shift_vol+signal_range[1]+1] - pre)
                dy = np.average(np.abs(y[shift_vol - signal_range[0]:shift_vol + signal_range[1] + 1] - pre))
                dy /= pre

            if np.isnan(mappa[ai, aj]):
                mappa[ai, aj] = dy
            else:
                mappa[ai, aj] += dy
            count[ai, aj] += 1

mappa[count > 0] /= count[count > 0]

mappa_full = np.copy(mappa)
mappa = funa.reduce_to_head(mappa)

# I want to look at the ratio of inhibitory to excitatory connections for different significance values
significance_numbers = [0.001, 0.005, 0.0075, 0.01, 0.05, 0.075, 0.1]
inhibitory_connections = []
total_connections = []
ratio = []
for sig in significance_numbers:
    Total_sig = (q_mat < sig).sum()
    total_connections.append(Total_sig)
    how_many_inhibit_and_sig = 0
    for ai in np.arange(len(q_mat)):
        for aj in np.arange(len(q_mat)):
            if mappa_full[ai, aj] < 0 and q_mat[ai, aj] < sig:
                how_many_inhibit_and_sig += 1
    inhibitory_connections.append(how_many_inhibit_and_sig)
    ratio.append(how_many_inhibit_and_sig/(Total_sig-how_many_inhibit_and_sig))


fig3 = plt.figure(figsize=(10, 5))
ax1 = fig3.add_subplot()
ax1.plot(significance_numbers, ratio, linestyle='--', marker='*')
ax1.set_xlabel("Significance Threshold")
ax1.set_ylabel("Inhibitory to Excitatory Ratio")
ax1.set_xscale("log")
#plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/LR_statistics.pdf")
plt.savefig("/projects/LEIFER/Sophie/Figures/Response_Statistics/Inhibitory_Excitatory.pdf")
fig3.clf()

q_sorted_inds = np.argsort(q)
q_sorted = q[q_sorted_inds]
mappa_flat = mappa_full[~np.isnan(q_mat)]
mappa_sorted = mappa_flat[q_sorted_inds]
signs = np.sign(mappa_sorted)
excitatory = [signs == 1][0]
inhibitory = [signs == -1][0]
cum_inh = np.cumsum(inhibitory)
cum_exc = np.cumsum(excitatory)
cum_total = cum_inh + cum_exc
ratio = cum_inh / cum_exc
fraction = cum_inh/cum_total

frac_at_sig = fraction[np.abs(q_sorted-significance_number).argmin()]

fig4 = plt.figure(figsize=(10, 5))
ax1 = fig4.add_subplot()
ax1.plot(q_sorted, ratio)
ax1.set_xlabel("Significance Threshold")
ax1.set_ylabel("Inhibitory to Excitatory Ratio")
#ax1.set_xscale("log")
#plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/LR_statistics.pdf")
plt.savefig("/projects/LEIFER/Sophie/Figures/Response_Statistics/Inhibitory_Excitatory_Ratio.pdf")
fig4.clf()

fig5 = plt.figure(figsize=(10, 5))
ax1 = fig5.add_subplot()
ax1.plot(q_sorted, fraction)
ax1.set_xlabel("Significance Threshold")
ax1.set_ylabel("Fraction of Inhibitory Responses")
#ax1.set_xscale("log")
#plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/LR_statistics.pdf")
plt.savefig("/projects/LEIFER/Sophie/Figures/Response_Statistics/Inhibitory_Fraction.pdf")
fig5.clf()

fig6 = plt.figure(figsize=(10, 5))
ax1 = fig6.add_subplot()
ax1.plot(q_sorted[np.where(q_sorted<0.1)], ratio[np.where(q_sorted<0.1)])
ax1.set_xlabel("Significance Threshold")
ax1.set_ylabel("Inhibitory to Excitatory Ratio")
plt.axvline(0.05)
#ax1.set_xscale("log")
#plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/LR_statistics.pdf")
plt.savefig("/projects/LEIFER/Sophie/Figures/Response_Statistics/Inhibitory_Excitatory_Ratio_lessthan.pdf")
fig6.clf()

fig7 = plt.figure(figsize=(5, 3))
ax1 = fig7.add_subplot()
ax1.plot(q_sorted[np.where(q_sorted<0.1)], fraction[np.where(q_sorted<0.1)], linewidth = 6)
ax1.set_xlabel("Significance Threshold", fontsize= 20)
ax1.set_ylabel("Fraction of Inhibitory Responses", fontsize= 20)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.set_xscale("log")
plt.axvline(0.05, color = "green", linewidth = 3)
plt.yticks(np.arange(0, 0.41, step=0.2), fontsize= 20)
plt.xticks(np.arange(0, 0.11, step=0.05), fontsize= 20)
#plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/LR_statistics.pdf")
plt.savefig("/projects/LEIFER/Sophie/Figures/Response_Statistics/Inhibitory_Fraction_lessthan.pdf", bbox_inches='tight')
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/Inhibitory_Fraction_lessthan.pdf", bbox_inches='tight')
fig7.clf()

print("done")

