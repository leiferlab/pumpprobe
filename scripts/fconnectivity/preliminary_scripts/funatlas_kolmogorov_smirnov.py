import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp
from multipy import fdr

ctrl = "--ctrl" in sys.argv
unc31 = "--unc31" in sys.argv
print("not ctrl",not ctrl)

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
if ctrl:
    ds_list = ["/projects/LEIFER/Sophie/NewRecordings/20220214/pumpprobe_20220214_155225/",
                "/projects/LEIFER/Sophie/NewRecordings/20220214/pumpprobe_20220214_171348/",
                "/projects/LEIFER/Sophie/NewRecordings/20220215/pumpprobe_20220215_112405/",
                "/projects/LEIFER/Sophie/NewRecordings/20220216/pumpprobe_20220216_161637/"]

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,                
                 "matchless_nan_th_from_file": True}
print(signal_kwargs)

if unc31:
    ds_tags = "unc31"
    strain = "unc31"
    ds_exclude_tags = None
else:
    ds_tags = None
    ds_exclude_tags = "mutant"
                 
funa = pp.Funatlas.from_datasets(
                           ds_list,
                           merge_bilateral=False,
                           merge_dorsoventral=False,
                           merge_numbered=False,signal="green",
                           signal_kwargs=signal_kwargs,
                           ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                           enforce_stim_crosscheck=not ctrl,
                           verbose=False)
                           

occ1_all, inclall_occ2 = funa.get_occurrence_matrix(inclall=True, req_auto_response=(not ctrl))

p, pdff, psd = funa.get_kolmogorov_smirnov_p(inclall_occ2,strain=strain)
print("funa method")
q = funa.get_kolmogorov_smirnov_q(inclall_occ2,strain=strain)
print("funa method")

p = funa.reduce_to_head(p)
pdff = funa.reduce_to_head(pdff)
psd = funa.reduce_to_head(psd)
occ1 = funa.reduce_to_head(occ1_all)

p[occ1==0]=np.nan

_, q = fdr.qvalue(p[np.isfinite(p)])
q_mat = np.copy(p)
q_mat[np.isfinite(p)] = q

fig = plt.figure(1)
ax = fig.add_subplot(111)
mappable=ax.imshow(p, vmin=0, vmax=.05)
fig.colorbar(mappable, ax=ax)

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.imshow(pdff)

fig = plt.figure(3)
ax = fig.add_subplot(111)
ax.imshow(psd)

fig=plt.figure(4)
ax = fig.add_subplot(111)
mappable2 = ax.imshow(q_mat, vmin=0, vmax=.1)
fig.colorbar(mappable2, ax=ax)

fig = plt.figure(5)
ax = fig.add_subplot(111)
ax.hist(np.ravel(p),bins=100,density=True)
ax.set_xlabel("Kolmogorov-Smirnov p"+(" ctrl" if ctrl else ""))

fig = plt.figure(6)
ax = fig.add_subplot(111)
ax.hist(np.ravel(pdff),bins=100,density=True)
ax.set_xlabel("Kolmogorov-Smirnov pdff"+(" ctrl" if ctrl else ""))

fig = plt.figure(7)
ax = fig.add_subplot(111)
ax.hist(np.ravel(psd),bins=100,density=True)
ax.set_xlabel("Kolmogorov-Smirnov psd"+(" ctrl" if ctrl else ""))

fig = plt.figure(8)
ax = fig.add_subplot(111)
ax.hist(np.ravel(q),bins=100,density=True)
ax.set_xlabel("Kolmogorov-Smirnov q"+(" ctrl" if ctrl else ""))

plt.show()

