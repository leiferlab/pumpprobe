import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

funa = pp.Funatlas(merge_bilateral=True)

escon = funa.get_esconn()
eescon = funa.get_effective_esconn()

funa.load_neuropeptide_expression_from_file()
funa.load_neuropeptide_receptor_expression_from_file()

y_npt = np.sum(funa.npt_exp_levels>2,axis=1)
y_nptr = np.sum(funa.nptr_exp_levels>2,axis=1)

not_pharynx_head_ai = np.array([ai for ai in funa.head_ai if ai not in funa.pharynx_ai])

fig = plt.figure(1,figsize=(14,6))
ax = fig.add_subplot(111)
ax.imshow(funa.npt_exp_levels[funa.pharynx_ai,:])
ax.set_xticks(np.arange(len(funa.npt_genes)))
ax.set_xticklabels(funa.npt_genes,fontsize=8,rotation=90)
ax.set_yticks(np.arange(len(funa.pharynx_ai)))
ax.set_yticklabels(funa.pharynx_ids)
ax.set_title("Neuropeptides")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/ph_npt_exp_map.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/ph_npt_exp_map.pdf",dpi=300,bbox_inches="tight")

fig = plt.figure(2,figsize=(14,6))
ax = fig.add_subplot(111)
ax.imshow(funa.nptr_exp_levels[funa.pharynx_ai,:])
ax.set_xticks(np.arange(len(funa.nptr_genes)))
ax.set_xticklabels(funa.nptr_genes,fontsize=8,rotation=90)
ax.set_yticks(np.arange(len(funa.pharynx_ai)))
ax.set_yticklabels(funa.pharynx_ids)
ax.set_title("Neuropeptide receptors")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/ph_nptr_exp_map.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/ph_nptr_exp_map.pdf",dpi=300,bbox_inches="tight")

fig = plt.figure(3)
ax = fig.add_subplot(111)
ax.plot(y_npt[funa.pharynx_ai])
ax.plot(y_nptr[funa.pharynx_ai])
ax.set_xticks(np.arange(len(funa.pharynx_ai)))
ax.set_xticklabels(funa.pharynx_ids)

fig = plt.figure(4)
ax = fig.add_subplot(121)
ynptA = y_npt[funa.pharynx_ai]
ynptA = ynptA[~(ynptA==0)]
ynptB = y_npt[not_pharynx_head_ai]
ynptB = ynptB[~(ynptB==0)]
ax.boxplot([ynptA,ynptB])
ax.set_xticks([1,2])
ax.set_xticklabels(["pharynx","not pharynx"])
ax.set_ylabel("neuropeptides/neuron")

ax = fig.add_subplot(122)
ynptrA = y_nptr[funa.pharynx_ai]
ynptrA = ynptrA[~(ynptrA==0)]
ynptrB = y_nptr[not_pharynx_head_ai]
ynptrB = ynptrB[~(ynptrB==0)]
ax.boxplot([ynptrA,ynptrB])
ax.set_xticks([1,2])
ax.set_xticklabels(["pharynx","not pharynx"])
ax.set_ylabel("neuropeptide receptors/neuron")

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/ph_npt_exp.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/ph_npt_exp.pdf",dpi=300,bbox_inches="tight")

fig = plt.figure(6)
ax = fig.add_subplot(111)
ax.imshow(escon[funa.pharynx_ai][:,funa.pharynx_ai])
ax.set_xticks(np.arange(len(funa.pharynx_ai)))
ax.set_xticklabels(funa.pharynx_ids)
ax.set_yticks(np.arange(len(funa.pharynx_ai)))
ax.set_yticklabels(funa.pharynx_ids)
ax.set_title("Direct extrasynaptic connections (Bentley et al.)")

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/ph_escon.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/ph_escon.pdf",dpi=300,bbox_inches="tight")

fig = plt.figure(7)
ax = fig.add_subplot(111)
ax.imshow(eescon[funa.pharynx_ai][:,funa.pharynx_ai])
ax.set_xticks(np.arange(len(funa.pharynx_ai)))
ax.set_xticklabels(funa.pharynx_ids)
ax.set_yticks(np.arange(len(funa.pharynx_ai)))
ax.set_yticklabels(funa.pharynx_ids)

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/ph_eff_escon.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/ph_eff_escon.pdf",dpi=300,bbox_inches="tight")

plt.show()
