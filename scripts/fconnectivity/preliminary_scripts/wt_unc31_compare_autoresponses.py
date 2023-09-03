import numpy as np, matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import pumpprobe as pp

plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.rc('axes',labelsize=18)

wt_f = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache.txt")
unc31_f = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_unc31.txt")

di = np.diag_indices(wt_f.shape[0])
a = wt_f[di]
a = a[~np.isnan(a)]
b = unc31_f[di]
b = b[~np.isnan(b)]

_,p_mw = mannwhitneyu(a,b,alternative="greater")
print(np.around(p_mw,2),"MW","that wt autoresponses are larger than unc-31")

fig = plt.figure(1)
ax = fig.add_subplot(111)
bins_y_a, bins, _ = ax.hist(a,alpha=0.3,color="C0",label="wt")
bins_y_b, bins, _ = ax.hist(b,bins=bins,alpha=0.3,color="C1",label="unc-31")
ax.plot((np.median(a),np.median(a)),(0,np.max(bins_y_a)),c="C0",alpha=0.5,label="wt median")
ax.plot((np.median(b),np.median(b)),(0,np.max(bins_y_b)),c="C1",alpha=0.5,label="unc-31 median")
ax.text(0.5*(np.median(a)+np.median(b)),np.max(bins_y_a)*1.1,pp.p_to_stars(p_mw),ha="center")
ax.set_ylim(0,np.max(bins_y_a)*1.2)
ax.set_xlabel(r"autoresponses $\Delta F/F$")
ax.set_ylabel("density")
ax.legend()
fig.tight_layout()
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/reviewers/wt_unc31_compare_autoresponses_A.txt",a)
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/reviewers/wt_unc31_compare_autoresponses_B.txt",b)

n1 = len(bins_y_a)
n2 = len(bins_y_b)
assert n1==n2
f = open("/projects/LEIFER/francesco/funatlas/figures/paper/reviewers/wt_unc31_compare_autoresponses.txt","w")
f.write("bin edge,wt,unc31\n")
s = ""
for i in np.arange(n1):
    s += str(bins[i])+","+str(bins_y_a[i])+","+str(bins_y_b[i])+"\n"
s = s[:-1]
f.write(s)
f.close()
    

fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/reviewers/wt_unc31_compare_autoresponses.pdf",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/reviewers/wt_unc31_compare_autoresponses.png",dpi=300,bbox_inches="tight")
plt.show()
