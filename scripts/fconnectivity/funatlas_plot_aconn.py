import numpy as np, sys, matplotlib.pyplot as plt
import pumpprobe as pp

head_only = "--head-only" in sys.argv

funa = pp.Funatlas(merge_bilateral=True,merge_dorsoventral=False,merge_AWC=True)
funa.load_aconnectome_from_file(chem_th=0,gap_th=0,exclude_white=False)


aconn_gap = funa.aconn_gap.copy()
aconn_chem = funa.aconn_chem.copy()
aconn = aconn_gap + aconn_chem
eaconn = funa.get_effective_aconn3(max_hops=2,gain_1=100)
aconn_n_hops = funa.get_n_hops_aconn(n_hops=3)

if head_only:
    aconn = aconn[funa.head_ai][:,funa.head_ai]
    aconn_gap = aconn_gap[funa.head_ai][:,funa.head_ai]
    aconn_chem = aconn_chem[funa.head_ai][:,funa.head_ai]
    eaconn = eaconn[funa.head_ai][:,funa.head_ai]
    aconn_n_hops = aconn_n_hops[funa.head_ai][:,funa.head_ai]

fig = plt.figure(1)
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(aconn,interpolation='none')
ax2.imshow(eaconn,interpolation='none')
ax3.imshow(aconn_n_hops,interpolation='none')
ax1.set_xlabel("from")
ax1.set_ylabel("to")
ax2.set_xlabel("from")
ax2.set_ylabel("to")
ax3.set_xlabel("from")
ax3.set_ylabel("to")

fig = plt.figure(2)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(aconn_gap,interpolation="none")
ax2.imshow(aconn_chem,interpolation="none")
ax1.set_xlabel("from")
ax1.set_ylabel("to")
ax1.set_title("Gap junctions")
ax2.set_xlabel("from")
ax2.set_ylabel("to")
ax2.set_title("Chemical synapses")

fig = plt.figure(3)
vmax = np.max(aconn)
ax1 = fig.add_subplot(111)
ax1.imshow(aconn,interpolation='none',cmap='coolwarm',vmax=vmax,vmin=-vmax)
ax1.set_xlabel("from")
ax1.set_ylabel("to")
ax1.set_title("Aconn")

plt.show()
