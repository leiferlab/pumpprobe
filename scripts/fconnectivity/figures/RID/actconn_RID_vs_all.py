import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

'''funa = pp.Funatlas(merge_bilateral=True)
RID, ADL, URX, AWB = funa.ai_to_head(funa.ids_to_i(["RID","ADL","URX","AWB"]))
act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_bilateral_merged.txt")'''

funa = pp.Funatlas(merge_bilateral=False)
RID, ADL, URX, AWB = funa.ai_to_head(funa.ids_to_i(["RID","ADLR","URXL","AWBL"]))
act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_no_merge.txt")

act_conn = funa.reduce_to_head(act_conn)

ondiag = np.zeros_like(act_conn,dtype=bool)
np.fill_diagonal(ondiag,True)
#excl = ondiag + np.isnan(act_conn)
#excl2 = np.isnan(act_conn[:,RID]) + np.arange(act_conn[:,RID].shape[0])==RID
excl = ondiag 
excl2 = np.arange(act_conn[:,RID].shape[0])==RID

fig = plt.figure(1)
ax = fig.add_subplot(111)
_,bins,_ = ax.hist(act_conn[~excl],bins=100,density=True,alpha=0.5,label="all->all")#range=(-1e-6,6e-3),
ax.hist(act_conn[:,RID][~excl2],bins=bins,density=True,alpha=0.5,label="RID->all",color="green")#range=(-1e-6,6e-3),
#bins = np.logspace(-11,np.log10(2.23),30)
#ax.hist(np.ravel(act_conn),bins=bins,density=True,alpha=0.5,label="all->all")
#ax.hist(act_conn[:,RID][~excl2],bins=bins,density=True,alpha=0.5,label="RID->all",color="green")
ax.axvline(act_conn[ADL,RID],color="k",ls="-",label="RID->ADLR")
ax.axvline(act_conn[AWB,RID],color="r",ls="--",label="RID->AWBL")
ax.axvline(act_conn[URX,RID],color="yellow",ls=":",label="RID->URXL")
ax.set_xlabel("Anatomy-derived responses (V)")
ax.set_ylabel("Density")
#ax.set_xscale("log")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(fontsize=14)
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/actconn_RID_vs_all.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig4/actconn_RID_vs_all.pdf",dpi=300,bbox_inches="tight")
plt.show()
