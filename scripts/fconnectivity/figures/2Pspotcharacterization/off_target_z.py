import numpy as np, sys
import matplotlib.pyplot as plt
import wormdatamodel as wormdm

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)

to_paper = "--to-paper" in sys.argv

folder = "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210209/pumpprobe_20210209_163726/"

rec = wormdm.data.recording(folder, rectype="2d")
events = rec.get_events()["optogenetics"]

x = (rec.T - rec.T[0])[:-3]
y = np.loadtxt(folder+"neuron.txt").T[1]
baseline = np.average(y[:10])
y -= baseline
y /= baseline

fig = plt.figure(1)
ax1 = fig.add_subplot(111)

ax1.plot(x,y)
ax1.axvspan(x[events["index"][0]],x[events["index"][0]]+events["properties"]["n_pulses"][0]/500000,alpha=0.3,color="k")
ax1.axvspan(x[events["index"][1]],x[events["index"][1]]+events["properties"]["n_pulses"][1]/500000,alpha=0.3,color="k")

#ax1.set_title(folder.split("/")[-2])
ax1.set_xlabel("Time (s)",fontsize=14)
ax1.set_ylabel("$\Delta$F/F",fontsize=14)

ax1.text(4.5, 1.13, "z+4 µm",fontsize=14,color="k",horizontalalignment='center',verticalalignment='center',weight="bold")
ax1.text(29., 1.13, "z+0 µm",fontsize=14,color="k",horizontalalignment='center',verticalalignment='center',weight="bold")

plt.tight_layout()
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/2Pspotcharacterization/off_target_z.png",dpi=300,bbox_inches="tight")
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/2Pspotcharacterization/off_target_z.svg",bbox_inches="tight")
if to_paper:
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS2/off_target_z.pdf",bbox_inches="tight")
plt.show()
