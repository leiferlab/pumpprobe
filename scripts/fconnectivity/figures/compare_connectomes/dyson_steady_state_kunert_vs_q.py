import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
                  
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": True}
                 

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=True,
                                 ds_tags=None,ds_exclude_tags="mutant",
                                 verbose=False)
                                 
# Get the qvalues
#occ1,occ2 = funa.get_occurrence_matrix(req_auto_response=True)
#occ3 = funa.get_observation_matrix(req_auto_response=True)
_,inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
q,p = funa.get_kolmogorov_smirnov_q(inclall_occ2,return_p=True)
qmax = np.nanmax(q)
q3 = 1.-q/qmax

# Prepare array to exclude elements on the diagonal
ondiag = np.zeros_like(q,dtype=bool)
np.fill_diagonal(ondiag,True)

actconn_inverted = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2_inverted/activity_connectome_bilateral_merged.txt")
actconn_inverted = np.abs(actconn_inverted)

excl = np.isnan(q)+np.isnan(actconn_inverted)+ondiag
r_q_actconn = np.corrcoef(q3[~excl],actconn_inverted[~excl])[0,1]
print("r_q_actconn",r_q_actconn)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.scatter(actconn_inverted[~excl],q[~excl],alpha=0.3)
ax3.set_xlabel("anatomy-derived effective weight after steady-stead fitting")
ax3.set_ylabel("q")
ax3.invert_yaxis()
fig3.tight_layout()

fig4 = plt.figure(4,figsize=(4,2))
ax = fig4.add_subplot(111)
bars = [r_q_actconn,
        ]
y = np.arange(len(bars))[::-1]/2
ax.barh(y,bars,height=0.4,align="center")
ax.set_xlim(0,0.5)
ax.set_xlabel("Correlation coefficient")
ax.set_yticks(y)
ax.set_yticklabels(["Fitted\nanatomical weights",
                    ],
                    rotation=0,va="center")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig4.tight_layout()
fig4.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/dyson_steady_state_kunert_bar_plot.png",dpi=300,bbox_inches="tight")
fig4.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/dyson_steady_state_kunert_bar_plot.png",dpi=300,bbox_inches="tight")
fig4.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/dyson_steady_state_kunert_bar_plot.pdf",dpi=300,bbox_inches="tight")

plt.show()
