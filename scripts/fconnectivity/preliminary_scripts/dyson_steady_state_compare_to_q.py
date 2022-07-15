import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

f = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_inverted.txt")
actconn_inverted = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2_inverted/activity_connectome_bilateral_merged.txt")

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,                 
                 "matchless_nan_th_from_file": True}
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 ds_tags=None,ds_exclude_tags="mutant",
                                 verbose=False)

_,inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
q,p = funa.get_kolmogorov_smirnov_q(inclall_occ2,return_p=True)

ondiag = np.zeros_like(q,dtype=bool)
np.fill_diagonal(ondiag,True)

excl = np.isnan(q) + ondiag
#r_f_q = np.corrcoef([np.ravel(f[~excl]),np.ravel(1.-q[~excl])])[0,1]

#print(r_f_q)
print(actconn_inverted.shape,p.shape) # ACTCONN IS MERGED BILATERALLY

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.scatter(actconn_inverted[~excl],p[~excl])
ax.invert_yaxis()
ax.set_xlabel("simulated response from anatomy with fitted weights")
ax.set_ylabel("p")
ax.set_xlim(-2e-5,2e-5)
plt.show()
