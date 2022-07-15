import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

funa = pp.Funatlas(merge_bilateral=True)
RID, ADL, URX, AWB = funa.ai_to_head(funa.ids_to_i(["RID","ADL","URX","AWB"]))

act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_bilateral_merged.txt")
act_conn = funa.reduce_to_head(act_conn)

ondiag = np.zeros_like(act_conn,dtype=bool)
np.fill_diagonal(ondiag,True)
excl = ondiag + np.isnan(act_conn)
excl2 = np.isnan(act_conn[:,RID]) + np.arange(act_conn[:,RID].shape[0])==RID

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.hist(act_conn[~excl],bins=100,range=(-1e-6,6e-3),density=True,alpha=0.5,label="all->all")
ax.hist(act_conn[:,RID][~excl2],bins=100,range=(-1e-6,6e-3),density=True,alpha=0.5,label="RID->all")
ax.axvline(act_conn[ADL,RID],color="k",ls="-",label="RID->ADL")
ax.axvline(act_conn[AWB,RID],color="r",ls="--",label="RID->AWB")
ax.axvline(act_conn[URX,RID],color="g",ls=":",label="RID->URX")
ax.set_xlabel("anatomy-derived responses (V)")
ax.set_ylabel("density")
ax.legend()
ax.set_yscale("log")
fig.tight_layout()
plt.show()
