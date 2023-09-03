import numpy as np, matplotlib.pyplot as plt

plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes',labelsize=18)

print("Plotting together the results of")
print("python kunert_vs_correlations_optimize_correlation_step2.py --no-merge --matchless-nan-th:0.5 --matchless-nan-th-added-only")
print("python funatlas_vs_correlations2.py --matchless-nan-th:0.5 --matchless-nan-th-added-only --no-merge")
print("python funatlas_vs_correlations_optimize_correlation.py --matchless-nan-th:0.5 --matchless-nan-th-added-only --no-merge")

r_kuncorr_spont_opt_top_n = "all"
r_ck_spont_opt_top_n = 6#13 
r_ck_spont_opt_top_n_unc31 = None

r_conn_spont = 0.02 #does not change with new wt pumpprobe datasets
r_kuncorr_spont = 0.12 #does not change with new wt pumpprobe datasets
r_kuncorr_spont_opt = 0.12 #does not change with new wt pumpprobe datasets
r_ck_spont = 0.21 #0.13 with only prerev datasets 
r_ck_spont_opt = 0.61 #0.50 with only prerev datasets 
#r_ck_spont_unc31 = 0.03 #does not change with new wt pumpprobe datasets
#r_ck_spont_opt_unc31 = np.nan #does not change with new wt pumpprobe datasets
r_stimcorr_spont = 0.23 
#r_stimcorr_spont_unc31 = 0.14 

fig = plt.figure(1,figsize=(12,5))
ax= fig.add_subplot(111)

dx = 0.7
ax.bar(0,r_conn_spont,width=0.38,color="C0")
ax.bar(1*dx,r_kuncorr_spont_opt,width=0.45,color="C9")
ax.bar(1*dx,r_kuncorr_spont,width=0.38,color="C0")
ax.bar(2*dx,r_ck_spont_opt,width=0.45,color="C9")
ax.bar(2*dx,r_ck_spont,width=0.38,color="C0")

ax.set_xticks(np.arange(3)*dx)
ax.set_xticklabels(["Anatomical\nweights","Anatomy-derived\ncorrelations\n(biophysical model)","Functionally-derived\ncorrelations"])#,"Raw stimulated\nactivity"])
ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])

ax.set_ylabel("Agreement with spontaneous\nactivity correlations\n(Pearson's corr coeff)")

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.tight_layout()
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/fig6/funatlas_vs_correlations_nth.txt",np.array([r_conn_spont,r_kuncorr_spont_opt,r_kuncorr_spont,r_ck_spont_opt,r_ck_spont]))
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig6/funatlas_vs_correlations_nth.pdf",dpi=300,bbox_inches="tight")
plt.show()
