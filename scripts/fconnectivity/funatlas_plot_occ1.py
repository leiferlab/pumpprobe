import numpy as np, sys, matplotlib.pyplot as plt
import pumpprobe as pp

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
to_paper = "--to-paper"
       
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    if sa[0] == "--ds-list": ds_list = sa[1]
        
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=False,
                                 merge_dorsoventral=False,merge_AWC=False,
                                 signal="green",signal_kwargs=signal_kwargs,
                                 ds_tags=None,ds_exclude_tags="mutant")

i_check = 0

##################################################
# Get response occurrence and observation matrices
##################################################
occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=True)
#occ3 = funa.get_observation_matrix(req_auto_response=True).astype(float)
occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=True).astype(float)
print(np.sum(np.any(funa.reduce_to_head(occ3),axis=1)), "n observed via occ3")
sens1, sens2 = funa.get_occurrence_matrix(req_auto_response=True,stim_shift=1)
############
# Run checks
############
diag = np.diag_indices(funa.n_neurons)
notocc3diag = occ3[diag]==0
print("CHECK_"+str(i_check)+": On the diagonal, np.all(occ1[occ3!=0]) is",
      np.all( occ1[diag][~notocc3diag]>0 ))
i_check += 1
print("CHECK_"+str(i_check)+": np.all(occ1[occ3==0]==0)",
      np.all(occ1[occ3==0]==0))
i_check += 1
this_check = True
for j in np.arange(funa.n_neurons):
    if np.any(np.delete(occ1[:,j],j)>0) and not occ1[j,j]>0:
        this_check=False
print("CHECK_"+str(i_check)+": No responses where no autoresponse:", this_check)
i_check += 1
############
# End checks
############
occ1_norm = occ1.astype(float)
occ1_norm[occ3>0] /= occ3[occ3>0]
occ1_norm[occ3==0] = 0

# Reduce to head
occ1 = funa.reduce_to_head(occ1)
occ1_norm = funa.reduce_to_head(occ1_norm)
sens1 = funa.reduce_to_head(sens1)

##########################################################################
# Get the arrays telling you how often a neuron is observed and stimulated
##########################################################################
njstim = funa.get_times_all_j_stimulated(req_auto_response=True)
njstim2 = funa.get_times_all_j_stimulated(req_auto_response=False)
niobs = funa.get_times_all_i_observed(req_auto_response=True)
# Get the argsort for the observation array
obsheadargsort = np.argsort(niobs[funa.head_ai])[::-1]
# Print which neurons are never stimulated or observed
missing_stim_head_neu = funa.head_ids[njstim[funa.head_ai]==0]
missing_obs_head_neu = funa.head_ids[niobs[funa.head_ai]==0]
print("Neurons in the head never stimulated: ", missing_stim_head_neu)
print("Neurons in the head never observed: ", missing_obs_head_neu)

#################################################
# Plot the matrices, and save some text files too
#################################################
fig = plt.figure(1,figsize=(9,9))
ax = fig.add_subplot(111)
# occ1 is the number of times i responds when j is stimulated (and autoresponds)
occ1_geq2 = np.copy(occ1)
occ1_geq2[occ1_geq2<2] = 0
ax.imshow(occ1_geq2)
ax.set_xlabel("from")
ax.set_ylabel("to")
ax.set_title("occ1 (head neurons) (occ1>2)")
ax.set_xticks(np.arange(len(funa.head_ids)))
ax.set_yticks(np.arange(len(funa.head_ids)))
ax.set_xticklabels(funa.head_ids,fontsize=5,rotation=90)
ax.set_yticklabels(funa.head_ids,fontsize=5)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/diagnostics/occ1.png",dpi=300,
            bbox_inches="tight")

fig = plt.figure(2,figsize=(16,9))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.imshow(occ1_norm*(occ1>=2))
ax1.set_xlabel("from")
ax1.set_ylabel("to")
ax1.set_title("occ1 normalized by occ3 (head neurons) (occ1>1)\n"+\
              "See CHECK_2 in terminal. If the upstream neuron is not\n"+\
              "observed/stim more than twice but the downstream neuron is\n"+\
              "labeled more than that, you can have nonzero off diagonal\n"+\
              "elements with zero diagonal elements")

thr = 0.75
occ1_norm2 = occ1_norm > thr
occ1_norm2 = occ1_norm2*(occ1>1)
ax2.imshow(occ1_norm2)
ax2.set_xlabel("from")
ax2.set_ylabel("to")
ax2.set_title("fconnected more than "+str(thr)+" of the times (occ1>1)")
# Save the pairs in occ1_norm2 to a textfile
lines = "# threshold: "+str(thr)+"\n"
for aij in np.array([np.where(occ1_norm2)[0],np.where(occ1_norm2)[1]]).T:
    lines += "from "+funa.head_ids[aij[1]] + " to "+funa.head_ids[aij[0]] + "\n"
lines = lines[:-1]
f = open("/projects/LEIFER/francesco/funatlas/diagnostics/"+\
         "connected_more_than_x_percent.txt","w")
f.write(lines)
f.close()

occ1_norm3 = occ1_norm < thr
ax3.imshow(occ1_norm3*(occ1>1))
ax3.set_xlabel("from")
ax3.set_ylabel("to")
ax3.set_title("fconnected less than "+str(thr)+" of the times (occ1>1)")

fig = plt.figure(3)
ax = fig.add_subplot(111)
ax.hist(niobs,density=False,range=(1,35),bins=30,label="i is obs",alpha=0.5)
ax.hist(njstim,density=False,range=(1,35),bins=30,label="j is stim",alpha=0.5)
ax.set_xlabel("times")
ax.set_title("observation and stimulation time\n(0 excluded)")
ax.legend()

fig = plt.figure(4,figsize=(16,9))
ax = fig.add_subplot(111)
x = np.arange(len(niobs[funa.head_ai]))
bars = niobs[funa.head_ai][obsheadargsort]
bars2 = njstim[funa.head_ai][obsheadargsort]
bars3 = njstim2[funa.head_ai][obsheadargsort]
labels = tick_label=funa.neuron_ids[funa.head_ai][obsheadargsort]
ax.bar(x,bars,alpha=0.5,label="observed")
ax.bar(x,bars3,alpha=0.5,label="stimulated (req_auto_response=False)")
ax.bar(x,bars2,alpha=0.5,label="stimulated")
ax.axhline(1,c='k',alpha=0.3)
ax.axhline(2,c='k',alpha=0.3)
ax.axhline(6,c='k',alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(labels,rotation=90,fontsize=6)
ax.set_yticks(np.append(ax.get_yticks(),[1,2,6]))
ax.legend()
plt.tight_layout()
plt.savefig("/projects/LEIFER/francesco/funatlas/diagnostics/"+\
            "neuron_hist_obs_stim.pdf", bbox_inches="tight")
            
fig = plt.figure(41,figsize=(16,9))
ax = fig.add_subplot(111)
x = np.arange(len(niobs[funa.head_ai]))
bars = niobs[funa.head_ai][obsheadargsort]
print(len(bars), "len bars")
print(np.sum(bars>0), "bars>0")
incons = np.logical_and(~np.any(occ3[funa.head_ai][:,funa.head_ai][obsheadargsort][:,obsheadargsort],axis=1),bars>0)
print(np.sum(incons),"incons")
print(funa.head_ids[obsheadargsort][incons])
bars2 = njstim[funa.head_ai][obsheadargsort]
bars3 = njstim2[funa.head_ai][obsheadargsort]
labels = tick_label=funa.neuron_ids[funa.head_ai][obsheadargsort]
ax.bar(x,bars,alpha=0.5,label="observed in n datasets")
ax.bar(x,bars2,alpha=0.5,label="stimulated n times")
ax.axhline(1,c='k',alpha=0.3)
ax.axhline(2,c='k',alpha=0.3)
ax.axhline(6,c='k',alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(labels,rotation=90,fontsize=6)
ax.set_yticks(np.append(ax.get_yticks(),[1,2,6]))
ax.set_ylabel("n")
ax.legend(fontsize=18)
plt.tight_layout()
plt.savefig("/projects/LEIFER/francesco/funatlas/diagnostics/"+\
            "neuron_hist_obs_stim_2.pdf", bbox_inches="tight")
if to_paper:
    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS6/neuron_hist_obs_stim_2_A.txt",np.array([x,bars]))
    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS6/neuron_hist_obs_stim_2_B.txt",np.array([x,bars2]))
    f = open("/projects/LEIFER/francesco/funatlas/figures/paper/figS6/neuron_hist_obs_stim_2.txt","w")
    f.write("neuron,observed,stimulated\n")
    s = ""
    n = len(labels)
    for i in np.arange(n):
        s += labels[i]+","+str(bars[i])+","+str(bars2[i])+"\n"
    s = s[:-1]
    f.write(s)
    f.close()
    ##FIXME plt.savefig(("/projects/LEIFER/francesco/funatlas/figures/paper/figS6/neuron_hist_obs_stim_2.pdf", bbox_inches="tight")
            
fig = plt.figure(5,figsize=(9,9))
ax = fig.add_subplot(111)
# occ1 is the number of times i responds when j is stimulated (and autoresponds)
ax.imshow(sens1)
ax.set_xlabel("from")
ax.set_ylabel("to")
ax.set_title("sens1 (head neurons)")
ax.set_xticks(np.arange(len(funa.head_ids)))
ax.set_yticks(np.arange(len(funa.head_ids)))
ax.set_xticklabels(funa.head_ids,fontsize=5,rotation=90)
ax.set_yticklabels(funa.head_ids,fontsize=5)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/diagnostics/sens1.png",dpi=300,
            bbox_inches="tight")

plt.show()
