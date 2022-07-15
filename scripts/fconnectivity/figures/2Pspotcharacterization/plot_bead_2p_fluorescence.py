import numpy as np
import matplotlib.pyplot as plt

folder = "/projects/LEIFER/francesco/pumpprobe/instrument_characterization/tuningthickness/20220222/stepbystep_20220222_093056/"

x = np.loadtxt(folder+"z.txt")[1:-2]
y = np.loadtxt(folder+"bead.txt").T[1]
y -= y[0]
y /= np.max(y)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1b = ax1.twiny()

ax1.plot(x,y)
#ax1.axhline(0,c="k",lw=0.5)
#ax1.axhline(0.5,c="k",lw=0.5)

#ax1b.plot(x/0.05,y,alpha=0)
#ax1b.axvline(-3.66,lw=0.5)
#ax1b.axvline(0.78,lw=0.5)

ax1.set_title(folder.split("/")[-2])
ax1.set_xlabel("Optog. etl focal power (dpt)")
ax1.set_ylabel("Intensity")
ax1b.set_xlabel("z (µm)")

plt.tight_layout()
plt.savefig("/".join(folder.split("/")[:-2])+"/"+folder.split("/")[-2]+"_bead.png",dpi=300,bbox_inches="tight")
plt.show()
