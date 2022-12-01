import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

plt.rc("xtick",labelsize=14)
plt.rc("ytick",labelsize=14)
plt.rc("axes",labelsize=14)

top_n_opt = 14
merge = "--no-merge" not in sys.argv
add_s = "" if merge else "_no_merge"
to_paper = "--to-paper" in sys.argv
shuffle = False
shuffle_n = 0
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv


for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    if sa[0] == "--shuffle-n": 
        shuffle_n = int(sa[1])
        shuffle = True
   
ds_list_spont = "/projects/LEIFER/francesco/funatlas_list_spont.txt"
                  
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}
                 
# Load Funatlas for spontaneous activity                                 
funa_spont = pp.Funatlas.from_datasets(ds_list_spont, merge_bilateral=merge,
                                         signal="green",
                                         signal_kwargs=signal_kwargs)
# Get the spontaneous activity correlation
spontcorr = funa_spont.get_signal_correlations()

# Prepare array to exclude elements on the diagonal
ondiag = np.zeros_like(spontcorr,dtype=bool); np.fill_diagonal(ondiag,True)

kuncorr = np.load("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_correlation_individual"+add_s+".npy")

'''if merge:
    funa_um = pp.Funatlas()
    qm = np.zeros(ondiag.shape)
    count = np.zeros_like(qm)
    for ai in np.arange(q.shape[0]):
        for aj in np.arange(q.shape[1]):
            ai2 = funa_spont.ids_to_i([funa_um.neuron_ids[ai]])
            aj2 = funa_spont.ids_to_i([funa_um.neuron_ids[aj]])
            
            if not np.isnan(q[ai,aj]):
                qm[ai2,aj2] += q[ai,aj]
                count[ai2,aj2] += 1

    qm /= count

    print((qm>0.5).shape)
    print(spontcorr.shape)
else:
    q = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q.txt")
    qm = q
    
#kuncorrcomplete = np.average(kuncorr,axis=0)
#excl = np.isnan(spontcorr) | np.isnan(kuncorrcomplete) | ondiag 
#excl = np.loadtxt("/projects/LEIFER/francesco/funatlas/excl_vs_correlations.txt").astype(bool)'''

# Sort the correlations with individual neurons being driven
all_r_spontcorr_kuncorr = np.zeros(funa_spont.n_neurons)
for ai in np.arange(funa_spont.n_neurons):
    print(ai,end="")
    
    excl = np.isnan(spontcorr) | np.isnan(kuncorr[ai]) | ondiag
    spontcorr_ = spontcorr[~excl]
    #Resymmetrize (has no effect)
    nanmask = np.isnan(kuncorr[ai])*np.isnan(kuncorr[ai].T)
    kuncorrai = 0.5*(np.nansum([kuncorr[ai],kuncorr[ai].T],axis=0))
    kuncorrai[nanmask] = np.nan
    kuncorr_ = kuncorrai[~excl]
    
    all_r_spontcorr_kuncorr[ai] = np.corrcoef([spontcorr_,kuncorr_])[0,1]
    print("\r",end="")
    
    
sorted_neurons = np.argsort(all_r_spontcorr_kuncorr)

if shuffle:
    shuffled_r = np.zeros(shuffle_n+1)
for shuffle_i in np.arange(shuffle_n+1):
    if shuffle and shuffle_i>0: np.random.shuffle(sorted_neurons)
    exclnan = np.isnan(all_r_spontcorr_kuncorr[sorted_neurons])
    nonnann = np.sum(~exclnan)
    all_top_neurons_r_spontcorr_kuncorr = np.zeros(nonnann)
    for top_n in np.arange(nonnann): ##[top_n_opt]:#
        print(top_n,end="")
        top_neurons = sorted_neurons[~exclnan][-top_n:]

        kuncorr_top_neurons = np.average(kuncorr[~exclnan][-top_n:],axis=0)
        #Resymmetrize (has no effect)
        nanmask = np.isnan(kuncorr_top_neurons)*np.isnan(kuncorr_top_neurons.T)
        kuncorr_top_neurons = 0.5*(np.nansum([kuncorr_top_neurons,kuncorr_top_neurons.T],axis=0))
        kuncorr_top_neurons[nanmask] = np.nan

        excl = np.isnan(spontcorr) | np.isnan(kuncorr_top_neurons) | ondiag
        spontcorr_ = spontcorr[~excl]
        kuncorr_ = kuncorr_top_neurons[~excl]
        top_neurons_r_spontcorr_kuncorr = np.corrcoef([spontcorr_,kuncorr_])[0,1]
        all_top_neurons_r_spontcorr_kuncorr[top_n] = top_neurons_r_spontcorr_kuncorr
        if shuffle: shuffled_r[shuffle_i] = top_neurons_r_spontcorr_kuncorr
        print("\r",end="")
        
        #print(top_neurons)
        #print([funa.neuron_ids[tn] for tn in top_neurons])
        #print(top_neurons_r_spontcorr_ck)
        if not shuffle and top_n == top_n_opt:
            fig = plt.figure(2)
            ax = fig.add_subplot(111)
            ax.plot(spontcorr_,kuncorr_,'o',alpha=0.3)
            ax.axvline(0,c="k",alpha=0.5)
            ax.axhline(0,c="k",alpha=0.5)
            ax.set_xlabel(r"$\langle$spontaneous correlation$\rangle_{ds}$")
            ax.set_ylabel("anatomy-derived correlation (top "+str(len(top_neurons))+" neurons)")
            fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kunert_vs_spont_scatter"+add_s+".png",dpi=300,bbox_inches="tight")
            fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS3/funatlas_vs_correlations_opt_corr_kunert_vs_spont_scatter"+add_s+".pdf",bbox_inches="tight")
            fig.tight_layout()
    
#print([funa_spont.neuron_ids[tn] for tn in top_neurons])
print("TAKE THIS FOR THE BAR PLOT top_n_opt=",top_n_opt," correlation",np.max(all_top_neurons_r_spontcorr_kuncorr),"funatlas_vs_correlation_combined_bar_plot.py")
print("OPT DRIVING NEURONS",[funa_spont.neuron_ids[tn] for tn in sorted_neurons[~exclnan][-top_n_opt:][::-1]])
    
if not shuffle:
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(all_r_spontcorr_kuncorr)
    fig.tight_layout()

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.plot(all_top_neurons_r_spontcorr_kuncorr)
    ax.set_xlabel("top n")
    ax.set_ylabel("r of anatomy-derived correlations\nand spontaneous correlations")
    fig.tight_layout()
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kunert_vs_spont"+add_s+".png",dpi=300,bbox_inches="tight")
    
if shuffle:
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    ax.hist(shuffled_r[1:],density=True,bins=30,label="shuffled")
    ax.axvline(shuffled_r[0],c="k",label="optimal")
    ax.set_xlabel("correlation coefficient of spontaneous correlations and\nanatomy-derived correlations with "+str(top_n_opt)+" driving neurons")
    ax.set_ylabel("density")
    fig.tight_layout()
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kunert_vs_spont_shuffled"+add_s+".png",dpi=300,bbox_inches="tight")


plt.show()
