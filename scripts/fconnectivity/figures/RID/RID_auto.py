import numpy as np, os, matplotlib.pyplot as plt
from matplotlib import cm
import pumpprobe as pp

q_th = 0.3

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=True,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="mutant",
                enforce_stim_crosscheck=False,verbose=False)
                
occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=False,inclall=True)
occ3 = funa.get_observation_matrix(req_auto_response=False)
qvalues = funa.get_kolmogorov_smirnov_q(occ2)

RID_exp_levels = funa.get_RID_downstream()
RID = funa.ids_to_i("RID")

# Print out list and save to file
sorter = np.argsort(qvalues[:,RID])
folder = os.path.abspath(os.path.dirname(__file__))+"/"
f = open(folder+"list.txt","w")
for i in np.arange(len(sorter)):
    s = sorter[i]
    if not np.isnan(qvalues[s,RID]):
        #print(i,np.around(qvalues[s,RID],3),funa.neuron_ids[s])
        f.write(str(i)+"\t"+str(np.around(qvalues[s,RID],3))+"\t"+funa.neuron_ids[s]+"\n")
f.close()
        
# Sort the neurons in different buckets, for the plot
RID_downs = RID_exp_levels>0
qvaluesnan = np.isnan(qvalues[:,RID])
maxqvalue = np.nanmax(qvalues[~qvaluesnan])

non_obs_downs = np.where(np.logical_and(qvaluesnan,RID_downs))[0]
non_obs_ndowns = np.where(np.logical_and(qvaluesnan,~RID_downs))[0]
# These should really be called obs_ instead of resp_
resp_downs = np.where(np.logical_and(~qvaluesnan,RID_downs))[0]
resp_ndowns = np.where(np.logical_and(~qvaluesnan,~RID_downs))[0]

fig = plt.figure(1,figsize=(5,5))
ax = fig.add_subplot(111)

ncol = 6
circlesize=20

# PLOT DOWNSTREAM OBSERVED
resp_downs_sorter = np.argsort(qvalues[resp_downs,RID])
ax.text(-0.25,-1+0.5,"Observed neurons",fontsize=10,weight="bold", va="bottom")
for j in np.arange(len(resp_downs)):
    i = resp_downs_sorter[j]
    col = j%ncol
    row = j//ncol
        
    lbl = funa.neuron_ids[resp_downs[i]]
    if lbl[-1] == "_": lbl=lbl[:-1]
    if lbl[-1] == "_": lbl=lbl[:-1]
    ftsz = 8
    if len(lbl)>3: ftsz = 6
    qval = qvalues[resp_downs[i],RID]
    
    ax.plot(col, row, 'o', markersize=circlesize, color=cm.viridis(qval/maxqvalue))
    ax.text(col, row, lbl, ha="center", va="center", color="white", fontsize=ftsz, weight="bold")
    
rows_resp_downs = row

ax.text(-0.25,rows_resp_downs+1+0.5,"Non-observed neurons",fontsize=10,weight="bold", va="bottom")
# PLOT DOWNSTREAM NOT OBSERVED
i = 0
for j in np.arange(len(non_obs_downs)):
    col = i%ncol
    row = i//ncol
    if non_obs_downs[j] not in funa.head_ai: continue
    i+=1
        
    lbl = funa.neuron_ids[non_obs_downs[j]]
    if lbl[-1] == "_": lbl=lbl[:-1]
    if lbl[-1] == "_": lbl=lbl[:-1]
    ftsz = 8
    if len(lbl)>3: ftsz = 6
    qval = qvalues[non_obs_downs[j],RID]
    
    ax.plot(col, row+rows_resp_downs+2, 'o', markersize=circlesize, color="gray")
    ax.text(col, row+rows_resp_downs+2, lbl, ha="center", va="center", color="white", fontsize=ftsz, weight="bold")
    

# PLOT NOT DOWNSTREAM OBSERVED
ax.text(ncol+1-0.25,-1+0.5,"Observed neurons",fontsize=10,weight="bold", va="bottom")
resp_ndowns_sorter = np.argsort(qvalues[resp_ndowns,RID])
for j in np.arange(len(resp_ndowns)):
    i = resp_ndowns_sorter[j]
    col = j%ncol
    row = j//ncol
        
    lbl = funa.neuron_ids[resp_ndowns[i]]
    if lbl[-1] == "_": lbl=lbl[:-1]
    if lbl[-1] == "_": lbl=lbl[:-1]
    ftsz = 8
    if len(lbl)>3: ftsz = 6
    qval = qvalues[resp_ndowns[i],RID]
    
    ax.plot(col+ncol+1, row, 'o', markersize=circlesize, color=cm.viridis(qval/maxqvalue))
    ax.text(col+ncol+1, row, lbl, ha="center", va="center", color="white", fontsize=ftsz, weight="bold")
    
rows_resp_ndowns = row

# PLOT NOT DOWNSTREAM NOT OBSERVED
ax.text(ncol+1-0.25,rows_resp_ndowns+1+0.5,"Non-observed neurons",fontsize=10,weight="bold", va="bottom")
i = 0
for j in np.arange(len(non_obs_ndowns)):
    col = i%ncol
    row = i//ncol
    if non_obs_ndowns[j] not in funa.head_ai: continue
    i+=1
        
    lbl = funa.neuron_ids[non_obs_ndowns[j]]
    if lbl[-1] == "_": lbl=lbl[:-1]
    if lbl[-1] == "_": lbl=lbl[:-1]
    ftsz = 8
    if len(lbl)>3: ftsz = 6
    qval = qvalues[non_obs_ndowns[j],RID]
    
    ax.plot(col+ncol+1, row+rows_resp_ndowns+2, 'o', markersize=circlesize, color="gray")
    ax.text(col+ncol+1, row+rows_resp_ndowns+2, lbl, ha="center", va="center", color="white", fontsize=ftsz, weight="bold")

ax.invert_yaxis()
plt.axis("off")
plt.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/RID_auto.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/RID_auto.svg",bbox_inches="tight")

# Make colorbar
fig = plt.figure(2,figsize=(1,5))
ax = fig.add_subplot(111)
pp.make_colorbar(ax,0,np.around(maxqvalue,1),0.1,"viridis")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/RID_auto_colorbar.png",dpi=300,bbox_inches="tight")
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/RID/RID_auto_colorbar.svg",bbox_inches="tight")

plt.show()
