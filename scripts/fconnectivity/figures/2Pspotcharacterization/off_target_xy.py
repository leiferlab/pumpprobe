import numpy as np, matplotlib.pyplot as plt, sys
import wormdatamodel as wormdm

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)

to_paper = "--to-paper" in sys.argv

im_fname='off_target_xy_image.png'
im=plt.imread(im_fname)


folder = "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211021/pumpprobe_20211021_112442/"
rec = wormdm.data.recording(folder)
e = rec.get_events()["optogenetics"]
ie = e["index"]

neu_i = 21

sig = wormdm.signal.Signal.from_file(folder,"green")
y = sig.get_smoothed(mode="sg_causal",n=13,poly=1)[:,neu_i]
baseline = np.average(y[:10])
y -= baseline
y /= baseline
x = np.arange(len(y))*rec.Dt

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(x,y)
ax.axvspan(ie[0]*rec.Dt,ie[0]*rec.Dt+e["properties"]["n_pulses"][0]/500000,alpha=0.3,color="k")
ax.axvspan(ie[1]*rec.Dt,ie[1]*rec.Dt+e["properties"]["n_pulses"][1]/500000,alpha=0.3,color="k")
ax.set_xlabel("time (s)",fontsize=14)
ax.set_ylabel("$\Delta$F/F",fontsize=14)
ax.text(3,1.25,"1",fontsize=14,color="k",horizontalalignment='center',verticalalignment='center',weight="bold")
ax.text(38,1.25,"2",fontsize=14,color="k",horizontalalignment='center',verticalalignment='center',weight="bold")

ax_i = ax.inset_axes((0.15,0.4,0.3,0.45),zorder=1)
ax_i.imshow(im)
ax_i.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

ax.text(16,0.63,"1",fontsize=14,color="y",horizontalalignment='center',verticalalignment='center',weight="bold")
ax.text(21.5,0.93,"2",fontsize=14,color="y",horizontalalignment='center',verticalalignment='center',weight="bold")

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/2Pspotcharacterization/off_target_xy.png",dpi=300,bbox_inches="tight") 
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/2Pspotcharacterization/off_target_xy.svg",bbox_inches="tight")
if to_paper:
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS2/off_target_xy.pdf",bbox_inches="tight")
plt.show()
