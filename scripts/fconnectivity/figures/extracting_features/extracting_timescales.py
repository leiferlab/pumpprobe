import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

plt.rc('axes',labelsize=16)

time = np.linspace(0,40,400)
ec = pp.ExponentialConvolution([0.1,0.12,3.])
y = ec.eval(time)
y_plt = np.zeros(len(y)+100)
y_plt[100:] = y
time2 = np.linspace(-10,40,500)

peak_t = ec.get_peak_time(time)
rise_t = ec.get_effective_rise_time(time)
decay_t = ec.get_effective_decay_time(time)

fig = plt.figure(1,figsize=(5,3))
ax = fig.add_subplot(111)
ax.axvline(peak_t,color="#bbbbbb")
ax.axvspan(peak_t-rise_t,peak_t,color="green",alpha=0.3)
ax.axvspan(peak_t,peak_t+decay_t,color="red",alpha=0.3)
ax.axhline(0,c="#333333")
ax.axhline(np.exp(-1)*np.max(y),c="#333333")
ax.plot(time2,y_plt,color="k")
ax.text(-3,0.037,"effective\nrise time",weight="bold",color="green",fontsize=16,horizontalalignment="center",verticalalignment="center")
ax.text(25,0.037,"effective decay\ntime",weight="bold",color="red",fontsize=16,horizontalalignment="center",verticalalignment="center")
ax.text(11,0.001,"baseline",weight="bold",color="#333333",fontsize=16)
ax.text(30,0.016,"peak e$^{-1}$",weight="bold",color="#333333",fontsize=16)
ax.axis("off")
ax.set_ylim(-0.005,)
ax.set_xlim(-10,)
ax.set_xlabel("time")

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/extracting_features/timescales.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/extracting_features/timescales.svg",bbox_inches="tight")
plt.show()
