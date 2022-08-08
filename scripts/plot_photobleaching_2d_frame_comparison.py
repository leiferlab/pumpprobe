import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mistofrutta as mf
import wormdatamodel as wormdm
import sys

volumes_1 = []
volumes_2 = []
folder_1 = None
folder_2 = None
xlim = []
ylim = []
dz_from_logbook_um = None
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--folder-1": 
        folder_1 = sa[1]
        if folder_1[-1]!="/":folder_1+="/"
    if sa[0] == "--folder-2": 
        folder_2 = sa[1]
        if folder_2[-1]!="/":folder_2+="/"
    if sa[0] == "--volumes-1": volumes_1 = [int(a) for a in sa[1].split(",")]
    if sa[0] == "--volumes-2": volumes_2 = [int(a) for a in sa[1].split(",")]
    if sa[0] == "--xlim": xlim = [int(a) for a in sa[1].split(",")]
    if sa[0] == "--ylim": ylim = [int(a) for a in sa[1].split(",")]
    if sa[0] == "--inset-limits-1": i1_00,i1_01,i1_10,i1_11 = [int(a) for a in sa[1].split(",")]
    if sa[0] == "--inset-limits-2": i2_00,i2_01,i2_10,i2_11 = [int(a) for a in sa[1].split(",")]
    if sa[0] == "--dz-from-logbook-um": dz_from_logbook_um = float(sa[1])
    
rec = wormdm.data.recording(folder_1)
print(len(rec.T),"volumes available")

rec.load(startVolume=volumes_1[0],nVolume=20)
frame_a = np.average(np.clip(rec.frame[:,0].astype(float),130,None),axis=0) #210
rec.load(startVolume=volumes_1[1],nVolume=20)
frame_b = np.average(np.clip(rec.frame[:,0].astype(float),130,None),axis=0)

frame_diff = (frame_b[:]-frame_a[:])#/frame_a[:]
frame_diff = -np.clip(frame_diff,None,0)
frame_diff /= np.max(np.abs(frame_diff))
frame_orig = frame_b[:]
frame_orig = frame_orig / np.max(frame_orig)

frame_rgb_1 = np.zeros((frame_orig.shape[0],frame_orig.shape[1],3))
frame_rgb_1[...,0] = np.power(frame_orig,0.3)
frame_rgb_1[...,1] = np.power(frame_diff,1.)
frame_rgb_1[...,2] = np.power(frame_diff,1.)

if folder_2 is not None:
    rec = wormdm.data.recording(folder_2)
    print(len(rec.T),"volumes available")

    rec.load(startVolume=volumes_2[0],nVolume=20)
    frame_a = np.average(np.clip(rec.frame[:,0].astype(float),130,None),axis=0)
    rec.load(startVolume=volumes_2[1],nVolume=20)
    frame_b = np.average(np.clip(rec.frame[:,0].astype(float),130,None),axis=0)

    #frame_diff = (frame_b[:-1]-frame_a[1:]) #This was for 20210420/pumpprobe_20210420_113239/
    frame_diff = (frame_b[:]-frame_a[:])
    frame_diff = -np.clip(frame_diff,None,0)
    frame_diff /= np.max(np.abs(frame_diff))
    #frame_orig = frame_b[:-1] #This was for 20210420/pumpprobe_20210420_113239/
    frame_orig = frame_b[:]
    frame_orig = frame_orig / np.max(frame_orig)

    frame_rgb_2 = np.zeros((frame_orig.shape[0],frame_orig.shape[1],3))
    frame_rgb_2[...,0] = np.power(frame_orig,0.3)
    frame_rgb_2[...,1] = np.power(frame_diff,1.)
    frame_rgb_2[...,2] = np.power(frame_diff,1.)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax2.imshow(frame_rgb_2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2b = ax2.inset_axes([0.5,0.7,0.2,0.2])
    frame_rgb_2_only_red = np.zeros_like(frame_rgb_2)
    frame_rgb_2_only_red[...,0] = frame_rgb_2[...,0]
    ax2b.imshow(frame_rgb_2_only_red[i2_00:i2_01,i2_10:i2_11],origin="lower")
    ax2b.set_xticks([])
    ax2b.set_yticks([])
    ax2b.spines['bottom'].set_color("white");ax2b.spines['top'].set_color("white");
    ax2b.spines['left'].set_color("white");ax2b.spines['right'].set_color("white")
    ax2c = ax2.inset_axes([0.75,0.7,0.2,0.2])
    ax2c.imshow(frame_rgb_2[i2_00:i2_01,i2_10:i2_11],origin="lower")
    ax2c.set_xticks([])
    ax2c.set_yticks([])
    ax2c.spines['bottom'].set_color("white");ax2c.spines['top'].set_color("white");
    ax2c.spines['left'].set_color("white");ax2c.spines['right'].set_color("white")
    rect = patches.Rectangle((i2_10, i2_00), i2_01-i2_00, i2_11-i2_10, linewidth=0.7, edgecolor='white', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title("Far from the objective (+ "+str(np.around(dz_from_logbook_um,1))+" µm)")
    #ax2.set_xlabel(folder_2.split("/")[-2])
    ax2.set_xlim(xlim[0],xlim[1])
    ax2.set_ylim(ylim[0],ylim[1])
else:
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    
ax1.imshow(frame_rgb_1)
ax1.set_xticks([])
ax1.set_yticks([])
ax1b = ax1.inset_axes([0.5,0.7,0.2,0.2])
frame_rgb_1_only_red = np.zeros_like(frame_rgb_1)
frame_rgb_1_only_red[...,0] = frame_rgb_1[...,0]
ax1b.imshow(frame_rgb_1_only_red[i1_00:i1_01,i1_10:i1_11],origin="lower")
ax1b.set_xticks([])
ax1b.set_yticks([])
ax1b.spines['bottom'].set_color("white");ax1b.spines['top'].set_color("white");
ax1b.spines['left'].set_color("white");ax1b.spines['right'].set_color("white")
ax1c = ax1.inset_axes([0.75,0.7,0.2,0.2])
ax1c.imshow(frame_rgb_1[i1_00:i1_01,i1_10:i1_11],origin="lower")#230:260,173:203
ax1c.set_xticks([])
ax1c.set_yticks([])
ax1c.spines['bottom'].set_color("white");ax1c.spines['top'].set_color("white");
ax1c.spines['left'].set_color("white");ax1c.spines['right'].set_color("white")
rect = patches.Rectangle((i1_10, i1_00), i1_01-i1_00, i1_11-i1_10, linewidth=0.7, edgecolor='white', facecolor='none')
ax1.add_patch(rect)
#ax1c.set_axis_off()
ax1.set_title("Close to the objective")
#ax1.set_xlabel(folder_1.split("/")[-2])
ax1.set_xlim(xlim[0],xlim[1])
ax1.set_ylim(ylim[0],ylim[1])
if folder_2 is not None:
    fname = folder_1.split("/")[-2]+"-"+folder_2.split("/")[-2]+".png"
else:
    fname = folder_1.split("/")[-2]+".png"
plt.tight_layout()
plt.savefig("/".join(folder_1.split("/")[:-2])+"/"+fname,dpi=300)
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS1/"+fname.split(".")[0]+".pdf",dpi=300,bbox_inches="tight")
plt.show()
quit()
