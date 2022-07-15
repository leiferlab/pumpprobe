import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes',labelsize=14)

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="mutant",
                verbose=False)
                
occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=False)
occ3 = funa.get_observation_matrix()

occ1_a = occ1>0
occ1_b = (occ1==0)*(occ3>0)
           
yes_g = []
no_g = []
count_g = []
for gap_th in np.arange(40):
    funa.load_aconnectome_from_file(chem_th=1e6,gap_th=gap_th,exclude_white=False)

    aconn = funa.get_boolean_aconn()

    yes_g.append(np.sum(aconn*occ1_a*(occ3>0))/np.sum(aconn*(occ3>0)))
    no_g.append(np.sum(aconn*occ1_b)/np.sum(aconn*(occ3>0)))
    count_g.append(np.sum(aconn*(occ3>0)))
    
yes_s = []
no_s = [] 
count_s = []    
for chem_th in np.arange(40):
    funa.load_aconnectome_from_file(chem_th=chem_th,gap_th=1e6,exclude_white=False)

    aconn = funa.get_boolean_aconn()

    yes_s.append(np.sum(aconn*occ1_a*(occ3>0))/np.sum(aconn*(occ3>0)))
    no_s.append(np.sum(aconn*occ1_b)/np.sum(aconn*(occ3>0)))
    count_s.append(np.sum(aconn*(occ3>0)))
 
fig = plt.figure(1)
ax = fig.add_subplot(111)
axb = ax.twinx()
ax.plot(yes_g,label="gap",c='C0')
#ax.plot(no_g,c='C0')
ax.plot(yes_s,label="chem",c='C1')
#ax.plot(no_s,c='C1')
ax.axhline(0.5,c="k")
ax.set_xlabel("threshold")
ax.set_ylabel("fraction captured in fconn")
axb.plot(count_g,c='C0',ls='--')
axb.plot(count_s,c='C1',ls='--')
axb.set_ylabel("number of connections above threshold")
ax.legend()
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/aconn_captured_in_fconn.png",dpi=300,bbox_inches="tight")
plt.show()
