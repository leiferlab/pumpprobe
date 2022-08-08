import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

bar_width = 0.4
d = 0.

int_map = np.loadtxt('/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache.txt')
q = np.loadtxt('/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q.txt')
int_map[q>0.05] = 0.0

funa = pp.Funatlas()

int_map = funa.reduce_to_head(int_map)
int_map,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(int_map,return_all=True)

exc = int_map > 0.1
inh = int_map < -0.1

tot_exc = np.sum(exc,axis=0)
tot_inh = np.sum(inh,axis=0)

sorter = np.argsort(tot_exc-tot_inh)[::-1]

fig = plt.figure(1,figsize=(14,6))
ax = fig.add_subplot(111)
x = np.arange(len(tot_exc))
bars1 = tot_exc[sorter]
bars2 = tot_inh[sorter]
ax.bar(x,bars1,color="red",width=bar_width,label="excitatory")
ax.bar(x+d,-bars2,color="blue",width=bar_width,label="inhibitory")

#ax.set_xlim(-1,lim)
ax.set_xticks(x)
ax.set_xticklabels(funa.head_ids[sorter_i][sorter],fontsize=6,rotation=90)

yticks = [-30,-20,-10,0,10,20,30]
ax.set_yticks(yticks)
ax.set_yticklabels([str(abs(yt)) for yt in yticks])
ax.set_ylabel("n",fontsize=18)
ax.legend(fontsize=18)
fig.tight_layout()

#fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS15/rank_exc_to_inh.pdf",dpi=300,bbox_inches="tight")
plt.show()
