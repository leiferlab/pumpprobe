import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

to_paper = "--to-paper" in sys.argv

folder_z = "/projects/LEIFER/francesco/pumpprobe/instrument_characterization/tuningthickness/20220222/stepbystep_20220222_093056/"

z_coord = np.loadtxt(folder_z+"z.txt")[1:-2]/0.05
z_coord -= 1.2
z = np.loadtxt(folder_z+"bead.txt").T[1] 
z -= z[0]
z /= np.max(z)

folder_xy = "/projects/LEIFER/francesco/pumpprobe/instrument_characterization/tuningthickness/20220222/objectivesregistration_20220222_090254/"

x = np.loadtxt(folder_xy+"spot_x.txt")
x -= x[0]
x /= np.max(x)
y = np.loadtxt(folder_xy+"spot_y.txt")
y -= y[0]
y /= np.max(y)

x_coord = np.arange(len(x))*0.42 - 9.2
y_coord = np.arange(len(y))*0.42 - 7.5

fig = plt.figure(1,figsize=(5,3))
ax = fig.add_subplot(111)
ax.plot(x_coord,x,label="x",lw=3)
ax.plot(y_coord,y,label="y",lw=3)
ax.plot(z_coord,z,label="z",lw=3)
ax.plot((-1.6,1.5),(0.5,0.5),ls='-',c="k")
ax.plot(-1.6,0.5,marker="<",c="k")
ax.plot(1.5,0.5,marker=">",c="k")
ax.text(0,0.4,"3.1 µm",horizontalalignment='center',fontsize=9,weight="bold")
ax.set_xlabel("position (µm)")
ax.set_ylabel("intensity (norm.)")
ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/2Pspotcharacterization/xyz.png",dpi=300,bbox_inches="tight")
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/2Pspotcharacterization/xyz.svg",bbox_inches="tight")
if to_paper:
    plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig1/xyz.pdf",bbox_inches="tight")
plt.show()
