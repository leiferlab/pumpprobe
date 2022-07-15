import numpy as np, matplotlib.pyplot as plt

plt.rc("xtick",labelsize=20)
plt.rc("ytick",labelsize=20)
plt.rc("axes",labelsize=20)

a = np.loadtxt("fpbase_spectra.csv").T
l = a[0]
ex = a[1]
em = a[2]

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(l,ex,c="C0",lw=4,label="absorption")
ax.plot(l,em,c="C2",lw=4,label="emission")
ax.axvline(505,c="cyan",lw=4,label="505 nm laser")
ax.set_xlabel("wavelength (µm)")
ax.set_ylabel("GCaMP spectra (arb. u.)")
ax.legend(fontsize=16)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/gur-3/GCaMP_exc.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/gur-3/GCaMP_exc.svg",bbox_inches="tight")
plt.show()
