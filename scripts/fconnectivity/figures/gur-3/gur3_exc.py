import numpy as np, matplotlib.pyplot as plt, sys

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes',labelsize=14)

to_paper = "--to-paper" in sys.argv

a = np.loadtxt("fpbase_spectra.csv").T
l = a[0]
ex = a[1]
em = a[2]

a = np.array([[0.9179,-10.5946],[0.9829,-10.5946],[1.0526,-10.5946],[1.1272,-10.5946],[1.2070,-10.5946],[1.2925,-10.5946],[1.3841,-10.5946],[1.4822,-10.5946],[1.5872,-10.5946],[1.6997,-10.5946],[1.8201,-10.5946],[1.9490,-10.5946],[2.5924,-10.4707],[2.7760,-10.5946],[2.9727,-10.5946],[3.1833,-10.5946],[3.4089,-10.5946],[3.6504,-10.5946],[3.9091,-10.5946],[4.1860,-10.5946],[4.4826,-10.5946],[4.8002,-10.4707],[5.1403,-9.2320],[5.5203,-7.0642],[6.2335,0.0338],[6.4948,12.5698],[6.7210,23.9662],[6.9550,35.9820],[7.2325,47.4846],[7.4620,56.6689],[7.7956,65.3649],[8.3004,72.5248],[8.8885,75.2500],[10.2085,77.5135]])

x,y = a.T
y -= y[0]
y /= 100

xp = np.linspace(x[0],x[-1],100)
yp = np.interp(xp,x,y)

fig = plt.figure(1,figsize=(3,2))
ax = fig.add_subplot(111)
ax.plot(xp,yp,c='C0')
ax.axvline(1.4,c="cyan")#c='#bbbbbb')
ax.set_xscale('log')
ax.set_xlabel("intensity (mW/mm$^2$)")
ax.set_ylabel("$\Delta$F/F")
#ax.set_ylabel("activation\n($\Delta$F/F)")
#ax.text(1.5,0.8,"imaging\nintensity",fontsize=12,weight="bold",verticalalignment="top")

axi = ax.inset_axes([2.0,0.3,3.5,0.6], transform=ax.transData)
axi.plot(l,ex,c="k")
axi.set_xlim(None,550)
axi.set_yticks([])
axi.set_xticks([505])
axi.set_xticklabels([r"$\lambda =$ 505 nm     "],fontsize=8,horizontalalignment="center")
axi.set_ylabel("GCaMP exc.",fontsize=8)
axi.axvline(505,c='cyan',lw=0.8)

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/gur-3/gu3_exc.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/gur-3/gu3_exc.svg",bbox_inches="tight")
if to_paper:
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig1/gu3_exc.pdf",bbox_inches="tight")
plt.show()
