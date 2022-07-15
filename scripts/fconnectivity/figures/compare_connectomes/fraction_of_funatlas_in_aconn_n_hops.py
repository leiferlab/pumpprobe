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

# TODO WITH QVALUES?

occ1_th = np.sort(np.ravel(occ1))[-50]
occ1_th = 0
fun = (occ1>occ1_th)

ths = [[0,0],[0,1e6],[1e6,0]]
titles = ["all","gap junction only","chemical synapses only"]
for i_th in np.arange(len(ths)):
    th = ths[i_th]
    funa.load_aconnectome_from_file(gap_th=th[0],chem_th=th[1],exclude_white=False)

    occ1_in_aconn = np.zeros(4)
    not_occ1_in_aconn = np.zeros(4)
    for i_h in np.arange(4):
        aconn = funa.get_effective_aconn2(i_h+1)
        
        occ1_in_aconn[i_h] = np.sum(np.logical_and(fun,aconn))/np.sum(fun)
        not_occ1_in_aconn[i_h] = np.sum(np.logical_and(~fun,aconn))/np.sum(~fun)
        
    w = 0.2
    fig = plt.figure(1+i_th,figsize=(5,3))
    ax = fig.add_subplot(111)
    ax.bar(np.arange(4)+1-w/2,occ1_in_aconn,w,label="functional and also in connectome")
    ax.bar(np.arange(4)+1+w/2,not_occ1_in_aconn,w,label="not functional but in connectome")
    ax.set_xticks([1,2,3,4])
    ax.set_xlabel("number of hops")
    ax.set_ylabel("fraction")
    ax.legend().get_frame().set_alpha(0.3)
    ax.set_title(titles[i_th]+" occ1_th="+str(occ1_th))
    ax.set_ylim(0,1)
    fig.tight_layout()
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/fconn_in_aconn_"+str(i_th)+"_occ1th_"+str(occ1_th)+".png",dpi=300,bbox_inches="tight")
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/fconn_in_aconn_"+str(i_th)+"_occ1th_"+str(occ1_th)+".svg",bbox_inches="tight")
plt.show()
