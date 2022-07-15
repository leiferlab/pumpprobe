import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,                 
                 "matchless_nan_th_from_file": True}
# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 ds_tags=None,ds_exclude_tags="mutant",
                                 #ds_tags="unc31",ds_exclude_tags=None,
                                 verbose=False)
_,inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
time = np.linspace(0,60,120)

dff = funa.get_max_deltaFoverF(inclall_occ2,time,mode="avg",nans_to_zero=True,normalize="none")
dff_std = funa.std_occ2(dff)
dff_avg = funa.average_occ2(dff)
dff_std = dff_std / dff_avg

n_aconn = 4
act_conn = []
for aconn_ds_i in np.arange(n_aconn):
    folder = "/projects/LEIFER/francesco/simulations/activity_connectome_sign2"+"_aconn-ds-"+str(aconn_ds_i)+"/"
    act_conn.append(np.loadtxt(folder+"activity_connectome.txt"))
    
act_conn = np.array(act_conn)

ac_avg = np.nanmean(act_conn,axis=0)

ac_std = np.nanstd(act_conn,axis=0)
ac_rstd = np.zeros_like(ac_std)*np.nan
ac_rstd[ac_avg!=0] = ac_std[ac_avg!=0]/ac_avg[ac_avg!=0]

ondiag = np.zeros_like(ac_avg,dtype=bool)
np.fill_diagonal(ondiag,True)
excl = np.isnan(ac_rstd)+ondiag

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.hist(ac_rstd[~excl],bins=500,range=(-2.5,2.5))
ax.set_xlabel("relative variability in anatomy-based simulation")
ax.set_ylim(0,115)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/variability_anatomy_vs_functional_hist_a.png",dpi=300,bbox_inches="tight")

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.plot(ac_avg[~excl],ac_rstd[~excl],'o',alpha=0.3)
ax.set_ylim(-2.5,2.5)
ax.set_xlabel("average value in anatomy-based simulation")
ax.set_ylabel("relative variability in anatomy-based simulation")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/variability_anatomy_vs_functional_2.png",dpi=300,bbox_inches="tight")


excl = np.isnan(ac_rstd)+np.isnan(dff_std)+ondiag
fig = plt.figure(3)
ax = fig.add_subplot(111)
ax.scatter(ac_rstd[~excl],dff_std[~excl],alpha=0.3)
ax.set_xlim(-2.5,2.5)
ax.set_ylim(-5,5)
ax.set_xlabel("relative variability in anatomy-based simulation")
ax.set_ylabel("relative variability in functional data")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/variability_anatomy_vs_functional_scatter.png",dpi=300,bbox_inches="tight")


excl1 = np.isnan(dff_std)+ondiag
excl2 = np.isnan(ac_rstd)+ondiag
fig = plt.figure(4)
ax = fig.add_subplot(111)
ax.hist(ac_rstd[~excl2],bins=500,range=(-2.5,2.5),label="anatomy",alpha=0.5)
ax.hist(dff_std[~excl1],bins=500,range=(-5,5),label="functional",alpha=0.5)
ax.set_xlabel("relative variability")
ax.set_ylim(0,115)
ax.legend()
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/variability_anatomy_vs_functional_hist_af.png",dpi=300,bbox_inches="tight")


plt.show()
