import matplotlib.pyplot as plt
import numpy as np

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

top_n_opt = 11

fig1 = plt.figure(1,figsize=(5,3))
ax = fig1.add_subplot(111)
bars = [0.38314186861671173,
        0.13747135048358577,
        ]
y = np.arange(len(bars))[::-1]/2
ax.barh(y,bars,height=0.4,align="center")
ax.set_xlim(0,0.5)
ax.set_xlabel("Correlation coefficient")
ax.set_yticks(y)
ax.set_yticklabels(["top "+str(top_n_opt)+"\nfunctional",
                    "top "+str(top_n_opt)+"\nanatomy",
                    ],
                    rotation=0,va="center")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig1.tight_layout()
fig1.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_correlations_opt_bar_plot.pdf",bbox_inches="tight")
plt.show()
