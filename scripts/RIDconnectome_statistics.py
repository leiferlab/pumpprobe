import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

plt.rc('axes',labelsize=12)
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = "unc31"
ds_exclude_tags = None
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


funa.load_extrasynaptic_connectome_from_file()
esconn_ma = funa.esconn_ma # Monoamines
esconn_np = funa.esconn_np # Neuropeptides
esconn = np.logical_or(esconn_ma,esconn_np)

#esconn_ma = funatlas.reduce_to_head(esconn_ma)
#esconn_np = funatlas.reduce_to_head(esconn_np)
#esconn = funatlas.reduce_to_head(esconn)

funa.load_aconnectome_from_file(chem_th=0, gap_th=0)
aconn_c = funa.aconn_chem
aconn_g = funa.aconn_gap
one_hop_aconn = funa.get_n_hops_aconn(1)


outgoing_g_all = np.sum(aconn_g, axis=0)
outgoing_c_all = np.sum(aconn_c, axis=0)
outgoing_anatom_connections_all = outgoing_c_all + outgoing_g_all
outgoing_e_all_number = np.sum(esconn, axis=0)
addedconn = aconn_c + aconn_g
bool_conn = addedconn > 0.0
outgoing_conn_all_number = np.sum(bool_conn, axis=0)

iids = jids = funa.neuron_ids
RIDind = np.where(iids == "RID")[0][0]
#np.sum(addedconn>0,axis=0)[RIDind]
outgoing_anatom_connections_RID = outgoing_anatom_connections_all[RIDind]
outgoing_anatom_number_connections_RID = outgoing_conn_all_number[RIDind]
outgoing_esconn_number_connections_RID = outgoing_e_all_number[RIDind]

iids_head = funa.head_ids
RIDind_head = np.where(iids_head == "RID")[0][0]
aconn_c_head = funa.reduce_to_head(aconn_c)
aconn_g_head = funa.reduce_to_head(aconn_g)
esconn_head = funa.reduce_to_head(esconn)

outgoing_g_head = np.sum(aconn_g_head, axis=0)
outgoing_c_head = np.sum(aconn_c_head, axis=0)
outgoing_anatom_connections_head= outgoing_c_head + outgoing_g_head

outgoing_e_head_number = np.sum(esconn_head, axis=0)
addedconn_head = aconn_c_head + aconn_g_head
bool_conn_head = addedconn_head > 0.0
outgoing_conn_head_number = np.sum(bool_conn_head, axis=0)

outgoing_anatom_connections_RID_head = outgoing_anatom_connections_head[RIDind_head]
outgoing_anatom_number_connections_RID_head = outgoing_conn_head_number[RIDind_head]
outgoing_esconn_number_connections_RID_head = outgoing_e_head_number[RIDind_head]

head_RID_extrasyn = funa.reduce_to_head(funa.get_RID_downstream())
num_RID_extrasyn = len(np.where(head_RID_extrasyn>0)[0])

fig2 = plt.figure(figsize=(3, 3))
#ax = fig.add_axes([0,0,1,1])
#probs = ['P(Connection)', 'P(Connection|Bilateral Pairs)']
labels = ['Extra-\nsynaptic', 'Synaptic']
RID_numbers = [num_RID_extrasyn, outgoing_anatom_number_connections_RID_head]
y_pos = range(len(labels))
ypos = [i for i in y_pos]
ax = plt.gca()
plt.ylabel("Predicted # \n of outgoing connections")
plt.bar(ypos, RID_numbers)
plt.xticks(ypos, labels, rotation=0)
plt.yticks(np.arange(0, 68, step=30))
#plt.yticks(np.arange(0, 0.5, step=0.2), fontsize= 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig2.tight_layout()
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/Outgoing_connections_RID_mod.pdf", bbox_inches='tight')
fig2.clf()

fig3, ax = plt.subplots(figsize=(6, 4))
n, bins, patches = ax.hist(outgoing_anatom_connections_head, 40, density=True, label='Occ3')
plt.axvline(outgoing_anatom_connections_RID_head, color = "maroon", linewidth = 3, linestyle = "dashed")
ax.set_xlabel('Number of Outgoing Synaptic Contacts', fontsize = 30)
ax.set_ylabel('PDF', fontsize = 30)
plt.yticks(np.arange(0, 0.011, step=0.005), fontsize= 20)
plt.xticks(np.arange(0, 801, step=400), fontsize= 20)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/Outgoing_syn_connections_head.pdf", bbox_inches='tight')
fig3.clf()


print("done")






                
               

