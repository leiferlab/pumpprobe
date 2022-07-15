import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

shuffle_connectome = False
shuffle_connectome_n = 1
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--shuffle-connectome": 
        shuffle_connectome = True
        shuffle_connectome_n = 1+int(sa[1])
    
plot = not shuffle_connectome_n > 1
if not plot: print("Skipping the plots because of the multiple shufflings.")

# DATASETS WITH STIMULATION OF RID
ds_list = [
    "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_101248/",
    "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_152524/",
    #"/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211104/pumpprobe_20211104_102437/" #WT
    ]
    
ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

funa = pp.Funatlas.from_datasets(
                ds_list,                
                merge_bilateral=True,merge_dorsoventral=False,merge_AWC=True,
                ds_exclude_tags="mutant")         

ai_RID = funa.ids_to_i("RID")

occ1, occ2 = funa.get_occurrence_matrix()
#occ3 = funa.get_observation_matrix()
dFF = funa.get_max_deltaFoverF(occ2,time=np.linspace(0,60,120))
dFF = funa.reduce_to_head(funa.average_occ2(dFF)[:,ai_RID])

fnames = ["../../preliminary_scripts/external_data/GenesExpressing-npr-4-thrs2.csv",
          "../../preliminary_scripts/external_data/GenesExpressing-npr-11-thrs2.csv",
          "../../preliminary_scripts/external_data/GenesExpressing-pdfr-1-thrs2.csv",
          #"external_data/GenesExpressing-daf-2-thrs2.csv"
          ]
          
trans_rec_pairs = ["FLP-14,NPR-4",
                   "FLP-14,NPR-11",
                   "PDF-1,PDFR-1",
                   #"INS-17,DAF-2"
                   ]
                   
trans_exp_level = np.array([110634.0,110634.0,157972.0,])#1505.0])

# Build expression levels                   
x = np.arange(len(funa.head_ai))
exp_levels = np.zeros((len(fnames),len(funa.head_ai)))
for i_f in np.arange(len(fnames)):
    f = open(fnames[i_f],"r")
    lines = f.readlines()
    f.close()
    
    exp_levels_ = np.zeros(funa.n_neurons)
    
    
    for line in lines[1:]:
        s = line.split(",")
        cell_id = s[1][1:-1]
        exp_level = float(s[2])
        
        cell_id = funa.cengen_ids_conversion(cell_id)
        
        for cid in cell_id:
            ai = funa.ids_to_i(cid)
            if ai<0 and shuffle_connectome_n==0: print(cid,"not found")
            exp_levels_[ai] = exp_level*trans_exp_level[i_f]
        
    if shuffle_connectome and i_shuffle>0:
        exp_levels_ = funa.shuffle_array(exp_levels_,shuffling_sorter)
    b = funa.reduce_to_head(exp_levels_)
    exp_levels[i_f] = b
    
c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fig = plt.figure(1)
ax = fig.add_subplot(221)
ax.plot(np.sum(exp_levels,axis=0)/np.max(np.sum(exp_levels,axis=0)),dFF,'o',c=c[0],label="all")
ax.set_title("all")
ax.set_xlabel("receptor exp levels (norm.)")
ax.set_ylabel("$<max|\Delta F/F|>$")
ax.ticklabel_format(style='plain')
ax.set_ylim(-0.1,1.1)
ax.legend()
for i in np.arange(len(trans_rec_pairs)):
    ax = fig.add_subplot(2,2,i+2)
    ax.plot(exp_levels[i]/np.max(exp_levels[i]),dFF,'o',c=c[i+1],label=trans_rec_pairs[i])
    ax.set_xlabel("receptor exp levels (norm.)")
    ax.set_ylabel("$<max|\Delta F/F|>$")
    ax.ticklabel_format(style='plain')
    ax.set_ylim(-0.1,1.1)
    ax.legend()
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/dFFvsexplevels.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/dFFvsexplevels.svg",bbox_inches="tight")

fig = plt.figure(2,figsize=(5,3))
ax = fig.add_subplot(111)
ax.set_xlabel("receptor exp levels (norm.)")
ax.set_ylabel("$<max|\Delta F/F|>$")
ax.ticklabel_format(style='plain')
for i in np.arange(len(trans_rec_pairs)):
    ax.plot(exp_levels[i]/np.max(exp_levels[i]),dFF,'o',c=c[i+1],label=trans_rec_pairs[i])
ax.set_ylim(-0.1,1.1)
ax.set_xticks([0,0.5,1.0])
ax.set_yticks([0,0.5,1.0])
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/dFFvsexplevels_single.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/dFFvsexplevels_single.svg",bbox_inches="tight")
plt.show()

