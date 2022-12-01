import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
from multipy import fdr

#this script is to print all the relevant numbers for the paper and save them as latex macros

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

enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv

# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,                 
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only, 
                 "photobl_appl":True}

funa = pp.Funatlas.from_datasets(
                ds_list,
                merge_bilateral=merge_bilateral,
                merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                signal_kwargs=signal_kwargs,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck,
                verbose=False)

number_of_animals = len(funa.ds_list)

print("number of animals used: " + str(number_of_animals))

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ,req_auto_response=req_auto_response)
# If occ2 needs to be filtered
#occ1,occ2 = funa.filter_occ12_from_sysargv(occ2,sys.argv)

#this gets the version of occ3 that has only the traces that pass the nan threshold
occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=req_auto_response)

inclall_occ2 = occ2


occ3_head = funa.reduce_to_head(occ3)

where_occ3_nonzero = np.where(occ3_head > 0)
number_pairs_measured_occ3 = len(where_occ3_nonzero[0])  #this is the number of pairs measured as in where occ3 is nonzero but I think we want the number of qvalues

#print("the number of pairs measured: " + str(number_pairs_measured))

record_from = np.sum(occ3_head, axis=1)
record_from_min_once_number = len(np.where(record_from > 0)[0])
percent_record_from_min_once = record_from_min_once_number/len(occ3_head)

stimed = np.sum(occ3_head, axis=0)
stimed_min_once_number = len(np.where(stimed > 0)[0])
percent_stimed_min_once = stimed_min_once_number/len(occ3_head)

max_occ3 = occ3_head.max()

print("max number of observations of a pair: " + str(max_occ3))


q_mat = funa.get_kolmogorov_smirnov_q(inclall_occ2)
print(np.sum(occ3!=0))
print(np.sum(~np.isnan(q_mat)))
qvalues_head = funa.reduce_to_head(q_mat)
q_head = qvalues_head[~np.isnan(qvalues_head)]

number_pairs_measured = len(q_head)

percent_pairs_measured = number_pairs_measured/(len(occ3_head)**2)

#print("the percent of pairs in the head measured: " + str(percent_pairs_measured))


# define the threshold for significance and calculate the total number of significant connections
significance_number = 0.05
Total_sig_head  =(q_head < significance_number).sum()
print("number of significant connections in the head: " + str(Total_sig_head))
percent_sig = Total_sig_head/number_pairs_measured
print("percent of significant connections in the head: " + str(percent_sig))

iids = jids = funa.neuron_ids  # gives ids for each global index




#first we want to look at the probability that two left right pairs are connected
LR_pairs_connected = 0
LR_pairs_notconnected = 0
LR_pairs_connected_head = 0
LR_pairs_notconnected_head = 0
headneurons = funa.head_ids
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

            elif jid == iid[0:-1] + "L" and qval > significance_number:
                LR_pairs_notconnected += 1
                if iid in headneurons:
                    LR_pairs_notconnected_head += 1

# here I am not double counting since a connection between i and j does not
# mean there has to be a connection between j and i
total_LR_pairs_head = LR_pairs_connected_head + LR_pairs_notconnected_head
P_connected_given_LR = LR_pairs_connected_head/total_LR_pairs_head
P_connected_total = Total_sig_head/len(q_head)

print("probability that a pair is connected: "+str(P_connected_total))
print("probability that a pair is connected given it is a bilateral pair: "+str(P_connected_given_LR))

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


q_sorted_inds = np.argsort(q_head)
q_sorted = q_head[q_sorted_inds]
mappa_flat = mappa[~np.isnan(qvalues_head)]
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

print("the fraction od inhibitory connections at " + str(significance_number) + " is: " + str(frac_at_sig))

funa.load_aconnectome_from_file(chem_th=0, gap_th=0)
one_hop_aconn = funa.get_n_hops_aconn(1)
one_hop_aconn_head = funa.reduce_to_head(one_hop_aconn)

funa.load_extrasynaptic_connectome_from_file()
esconn_ma = funa.esconn_ma # Monoamines
esconn_np = funa.esconn_np # Neuropeptides
esconn = np.logical_or(esconn_ma,esconn_np)

esconn_head = funa.reduce_to_head(esconn)
extrasyn_connections_head = np.where(esconn_head == True)
number_extrasyn_connections = len(np.where(esconn_head == True)[0])
total_possible_connections_head = len(esconn_head)**2
prob_esconn = number_extrasyn_connections/total_possible_connections_head

pharynx_parts = funa.ganglia["pharynx"]
pharynx_neurons = funa.ganglia[pharynx_parts[0]] + funa.ganglia[pharynx_parts[1]]
pharynx_ids = []
for neuron in pharynx_neurons:
    pharynx_ids.append(np.where(funa.head_ids == neuron)[0][0])

esconn_parynx = esconn_head[pharynx_ids][:,pharynx_ids]
extrasyn_connections_pharynx = np.where(esconn_parynx == True)
number_extrasyn_connections_pharynx = len(extrasyn_connections_pharynx[0])
total_possible_connections_pharynx = len(esconn_parynx)**2
prob_esconn_pharynx = number_extrasyn_connections_pharynx/total_possible_connections_pharynx

syn_connections_head = np.where(one_hop_aconn_head == True)
number_syn_connections = len(syn_connections_head[0])
prob_aconn = number_syn_connections/total_possible_connections_head

aconn_parynx = one_hop_aconn_head[pharynx_ids][:,pharynx_ids]
syn_connections_pharynx = np.where(aconn_parynx == True)
number_syn_connections_pharynx = len(syn_connections_pharynx[0])
prob_aconn_pharynx = number_syn_connections_pharynx/total_possible_connections_pharynx

frac_esconn_head = number_extrasyn_connections/(number_extrasyn_connections+number_syn_connections)
frac_esconn_pharynx = number_extrasyn_connections_pharynx/(number_extrasyn_connections_pharynx+number_syn_connections_pharynx)

#with open('/projects/LEIFER/Sophie/Numbers_for_paper.txt', 'w') as f:
with open('/projects/LEIFER/francesco/funatlas/figures/paper/Numbers_for_paper.txt', 'w') as f:
    f.write(r"\newcommand{\dataNumAnimals}{"+str(number_of_animals)+" }")
    f.write("\n")
    f.write(r"\newcommand{\dataNumPairsMeasured}{" + str(number_pairs_measured) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataMaxTimesMeasured}{" + str(max_occ3) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataPercentPairsMeasured}{" + str(int(round(percent_pairs_measured*100, 0))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataNumSigPairs}{" + str(Total_sig_head) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataPercentSigPairs}{" + str(int(round(percent_sig*100, 0))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataProbConnectedTotal}{" + str(int(round(P_connected_total, 2))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataProbConnectedBilateral}{" + str(int(round(P_connected_given_LR, 2))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataFracInhibitory}{" + str(int(round(frac_at_sig*100,2))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataProbAconnAll}{" + str(int(round(prob_aconn*100, 0))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataProbAconnPharynx}{" + str(int(round(prob_aconn_pharynx * 100, 0))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataPercentRecordMinOnce}{" + str(int(round(percent_record_from_min_once * 100, 0))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataNumberRecordMinOnce}{" + str(record_from_min_once_number) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataNumberStimMinOnce}{" + str(stimed_min_once_number) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataPercentStimMinOnce}{" + str(int(round(percent_stimed_min_once * 100, 0))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataFracEsconnHead}{" + str(int(round(frac_esconn_head * 100, 0))) + " }")
    f.write("\n")
    f.write(r"\newcommand{\dataFracEsconnPharynx}{" + str(int(round(frac_esconn_pharynx * 100, 0))) + " }")

#calculate the number of neurons that we have recorded from ie rows in qvalues (or maybe occ3) of which there is at least one box that has nonzero value
#change percents



print("done")
