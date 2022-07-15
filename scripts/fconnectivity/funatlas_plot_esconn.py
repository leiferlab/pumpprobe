import numpy as np, sys, matplotlib.pyplot as plt
import pumpprobe as pp

head_only = "--head-only" in sys.argv

funa = pp.Funatlas(merge_bilateral=True,merge_dorsoventral=False,merge_AWC=True)
funa.load_extrasynaptic_connectome_from_file()

esconn_ma = funa.esconn_ma # Monoamines
esconn_np = funa.esconn_np # Neuropeptides
esconn = np.logical_or(esconn_ma,esconn_np)

if head_only:
    esconn_ma = funa.reduce_to_head(esconn_ma)
    esconn_np = funa.reduce_to_head(esconn_np)
    esconn = funa.reduce_to_head(esconn)

fig = plt.figure(1,figsize=(16,9))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(esconn_ma,interpolation='none')
ax2.imshow(esconn_np,interpolation='none')
ax3.imshow(esconn,interpolation='none')
ax1.set_xlabel("from")
ax1.set_ylabel("to")
ax1.set_title("Monoamines")
ax2.set_xlabel("from")
ax2.set_ylabel("to")
ax2.set_title("Neuropeptides")
ax3.set_xlabel("from")
ax3.set_ylabel("to")
ax3.set_title("Monoamines and neuropeptides")
plt.tight_layout()
plt.show()
