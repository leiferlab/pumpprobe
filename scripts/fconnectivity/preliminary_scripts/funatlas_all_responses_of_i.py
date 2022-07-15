import numpy as np, matplotlib.pyplot as plt, sys, os
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

dst = None
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--dst": dst  = sa[1]

if dst is None: print("dst is None");quit()

exclude_j = ["AVH_","AVJ_","FLP_","I1_","IL1V_","OLQD_","RMDV_","RME_",]

cs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

fig = plt.figure(1)
ax1 = fig.add_subplot(111)

file_list = os.listdir(dst)

for i in np.arange(len(file_list)):
    if file_list[i][-14:] == "_ys_interp.npy":
        j_id = file_list[i].split("->")[0]
        if j_id in exclude_j: continue
        a = np.load(dst+file_list[i])
        time = a[0]
        ys = a[1:]
        
        confidences = np.load(dst+file_list[i][:-14]+"_confidences.npy")
        
        y_avg = np.sum(ys*confidences[:,None],axis=0)/np.sum(confidences)
        
        ax1.plot(time,y_avg/np.max(y_avg),label=j_id,c=cs[(i-3)%len(cs)])

ax1.axvline(0,c="k",alpha=0.5)
ax1.set_xlabel("Time (s)",fontsize=14)
ax1.set_ylabel("Activity (normalized)", fontsize=14)
ax1.set_title("Average responses of targeted neurons")
ax1.legend(loc=2,ncol=2)
plt.tight_layout()
plt.savefig(dst+"0_auto.png",dpi=150)
plt.show()
