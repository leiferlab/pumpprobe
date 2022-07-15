import sys
import numpy as np
import mistofrutta as mf
import wormdatamodel as wormdm
import wormbrain as wormb
import matplotlib.pyplot as plt
import pumpprobe as pp
import savitzkygolay as sg

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

folder = sys.argv[1]
if folder[-1]!="/": folder += "/"

alphabetical_sort = "--alphabetical-sort" in sys.argv
labels_both_sides = "--labels-both-sides" in sys.argv
plot_bars = "--no-bars" not in sys.argv
only_labeled = "--only-labeled" in sys.argv
show_stim_index = "--show-stim-index" in sys.argv
hide_axes_spines = "--hide-axes-spines" in sys.argv
to_paper = "--to-paper" in sys.argv

nan_thresh = 0.8 #0.3 0.5
signal = "green"
for s in sys.argv:
    sa = s.split(":")
    if sa[0]=="--nan-th": nan_thresh = float(sa[1])
    if sa[0]=="--signal": signal = sa[1]

rec = wormdm.data.recording(folder)
events = rec.get_events()['optogenetics']
#ratio = wormdm.signal.Signal.from_signal_and_reference(folder,"green","red",method=0.0)
sig = wormdm.signal.Signal.from_file(folder,signal,matchless_nan_th_from_file=True)
ratio = sig
ratio.remove_spikes()
ratio.appl_photobl()
#ratio.median_filter();ratio.median_filter();ratio.median_filter();
cerv = wormb.Brains.from_file(folder)
ref_index = ratio.info['ref_index']
labels = cerv.get_labels(ref_index,attr=True)
fconn = pp.Fconn.from_file(folder)

fig = plt.figure(1,figsize=(10,7))
ax1 = fig.add_subplot(111)
Delta = 5.
time = np.arange(ratio.data.shape[0])*rec.Dt

j=0
plotted = []
baseline = []
colors = []
Y = []

argsort_labels = np.argsort(labels)[::-1]

for ip in np.arange(ratio.data.shape[1]):
    if alphabetical_sort:
        i = argsort_labels[ip]
    else:
        i = ip
        
    y = ratio.data[:,i]
    loc_std = np.sqrt(np.nanmedian(np.nanvar(rolling_window(y, 8), axis=-1)))
    
    too_many_nans = np.sum(ratio.nan_mask[:,i]) > nan_thresh*ratio.data.shape[0]

    if True:#loc_std<50. and loc_std>0.01 and not too_many_nans: #50 0.01 # 10 
        #tot_std = np.std(y)
        #spikes = np.where(y-np.average(y)>tot_std*5)[0]
        #for spike in spikes:
        #    y[spike] = y[spike-1]
        #y = ratio.get_smoothed(127,ip,3,"sg_causal")
        y = ratio.get_smoothed(13,i,1,"sg_causal") #13 i 3 sg
        y /= loc_std
        
        if labels[i] == "" and only_labeled: continue
        if labels[i] == "AMsoL" or labels[i] == "AMsoR" or labels[i] == "AMso" or labels[i] == "AMSoL"or labels[i] == "AMSoR"or labels[i] == "AmSo": continue
        
        DD = j*Delta-np.median(np.sort(y)[:100])
        l, = ax1.plot(time,y+DD,lw=0.8)
        ax1.annotate(labels[i],(-150-90*(j%2),j*Delta),c=l.get_color(),fontsize=8)
        if labels_both_sides:
            ax1.annotate(labels[i],(time[-1]+20+70*(j%2),j*Delta),c=l.get_color(),fontsize=8)
        plotted.append(i)
        baseline.append(DD)
        colors.append(l.get_color())
        Y.append(y)
        j += 1
        
for ie in np.arange(len(events['index'])):
    e = events['index'][ie]
    i_targeted = fconn.stim_neurons[ie]
    lbl_targeted = labels[i_targeted]
    
    colorjp = 'k'
    if i_targeted in plotted:
        jp = plotted.index(i_targeted)
        colorjp=colors[jp]
        
        if plot_bars:
            bly = Y[jp][e] + baseline[jp]
            ax1.plot((e*rec.Dt,e*rec.Dt),(bly,bly+Delta),c=colorjp, lw=2)
    
    if fconn.stim_neurons[ie] not in [-1,-2] or show_stim_index:
        if show_stim_index and lbl_targeted=="": lbl_targeted = str(ie)
        ax1.axvline(e*rec.Dt,c='k',alpha=0.1,lw=1)
        ax1.annotate(lbl_targeted,(e*rec.Dt,(j+1)*Delta),c=colorjp,fontsize=6,rotation=45)
    
    
        

if labels_both_sides:
    ax1.set_xlim(-270,time[-1]+270)
else:
    ax1.set_xlim(-270,time[-1])
ax1.set_ylim(-Delta,Delta*(j+6))
ax1.set_yticks([])
ax1.set_xlabel("Time (s)",fontsize=14)
ax1.set_ylabel("GCaMP fluorescence",fontsize=14)
if hide_axes_spines:
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_visible(False)

try:
    plt.savefig(folder+"responses/"+folder.split("/")[-2]+"_waterfall.png", dpi=300, bbox_inches="tight",metadata={"Comment":" ".join(sys.argv)})
    plt.savefig(folder+"responses/"+folder.split("/")[-2]+"_waterfall.pdf", dpi=300, bbox_inches="tight")
    if to_paper:
        plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig1/"+folder.split("/")[-2]+"_waterfall.pdf", dpi=300, bbox_inches="tight")
except:
    print("Figures not saved")
plt.show()
