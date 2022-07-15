import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

q_th = 0.3
for s in sys.argv:
    sa = s.split(":")
    if sa[0]=="--q-th": q_th=float(sa[1])

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes',labelsize=14)

act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_bilateral_merged.txt")
act_conn[np.isnan(act_conn)] = 0.0

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="mutant",
                verbose=False)
                
_, inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
qvalues = funa.get_kolmogorov_smirnov_q(inclall_occ2)
incl = ~np.isnan(qvalues)

q = qvalues[incl]<=q_th

sparseness = np.sum(q)/np.prod(q.shape)
th = pp.Funatlas.threshold_to_sparseness(act_conn,sparseness)
act_conn2 = (np.abs(act_conn)>=th)[incl]

tot_act_conn2 = np.sum(act_conn2)
frac = np.sum(np.logical_and(q,act_conn2))/tot_act_conn2

print("Target sparseness:",sparseness)
print("Activity connectome threshold:",th)
print("Number of connections in the sparsified matrix:",tot_act_conn2)
print("Fraction of sparsified activity connectome for which we have a q<=",q_th,":",frac)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(act_conn[incl],qvalues[incl],'o')
ax.axvline(th)
ax.axhline(q_th)
plt.show()
