import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

plt.rc("xtick",labelsize=14)
plt.rc("ytick",labelsize=14)
plt.rc("axes",labelsize=14)

top_n_opt = 14
merge = "--no-merge" not in sys.argv
print("merge:",merge)
to_paper = "--to-paper" in sys.argv
unc31 = "--unc31" in sys.argv
strain = "" if not unc31 else "unc31"
ds_tags = None if not unc31 else "unc31"
ds_exclude_tags = "mutant" if not unc31 else None
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
shuffle = False
shuffle_k = False
shuffle_n = 0
shuffle_k_n = 0
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    if sa[0] == "--shuffle-n": 
        shuffle_n = int(sa[1])
        shuffle = True
    if sa[0] == "--shuffle-k-n":
        shuffle_k_n = int(sa[1])
        shuffle_k = True
   

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_list_spont = "/projects/LEIFER/francesco/funatlas_list_spont.txt"
                  
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,    
                 "photobl_appl":True,               
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=merge,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                                 enforce_stim_crosscheck=True,
                                 verbose=False)
                                 
# Load Funatlas for spontaneous activity                                 
funa_spont = pp.Funatlas.from_datasets(ds_list_spont, merge_bilateral=merge,
                                         signal="green",signal_kwargs=signal_kwargs)
# Get the spontaneous activity correlation
spontcorr = funa_spont.get_signal_correlations()

# Get the qvalues
occ1,occ2 = funa.get_occurrence_matrix(req_auto_response=True)
occ3 = funa.get_observation_matrix(req_auto_response=True)
_,inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
q,p = funa.get_kolmogorov_smirnov_q(inclall_occ2,return_p=True,strain=strain)

# Prepare array to exclude elements on the diagonal
ondiag = np.zeros_like(q,dtype=bool)
np.fill_diagonal(ondiag,True)

km = funa.get_kernels_map(occ2,occ3,filtered=True,include_flat_kernels=True)
km[q>0.05]=np.nan

###FIXME excl_ = np.loadtxt("/projects/LEIFER/francesco/funatlas/excl_vs_correlations.txt").astype(bool)
#print("\n\n\n\n\n\nUSING EXCL FROM FILE\n\n\n\n\n\n")

if shuffle_k:
    shuffled_k_r = []
    shuffled_k_r_max = np.zeros(shuffle_k_n+1)
for shuffle_k_i in np.arange(shuffle_k_n+1):
        
    if shuffle_k and shuffle_k_i>0:
        sorter_i = np.random.permutation(km.shape[0])
        sorter_j = np.random.permutation(km.shape[1])
        km = km[sorter_i]
        km = km[:,sorter_j]
    
    # Get the kernel-derived correlations
    all_ck = np.zeros((funa.n_neurons,funa.n_neurons,funa.n_neurons))
    all_r_spontcorr_ck = np.zeros(funa.n_neurons)
    
    for ai_i in np.arange(funa.n_neurons):
        print(ai_i,end="")
        all_ck[ai_i] = funa.get_correlation_from_kernels_map(km,occ3,js=[ai_i],set_unknown_to_zero=False)#True)
        all_ck[ai_i][q>0.05]=np.nan
        nanmask = np.isnan(all_ck[ai_i])*np.isnan(all_ck[ai_i].T)
        all_ck[ai_i] = 0.5*(np.nansum([all_ck[ai_i],all_ck[ai_i].T],axis=0))
        all_ck[ai_i][nanmask] = np.nan
        
        excl = np.isnan(spontcorr) | np.isnan(all_ck[ai_i]) | ondiag 
        ##FIXME excl = excl_ | np.isnan(all_ck[ai_i])
        spontcorr_ = spontcorr[~excl]
        ck_ = all_ck[ai_i][~excl]
        all_r_spontcorr_ck[ai_i] = np.corrcoef([spontcorr_,ck_])[0,1]
        #if uncommenting bring excl up
        #excl = np.isnan(spontcorr)+np.isnan(all_ck[ai_i])+ondiag+np.isnan(q)
        #weights_= 1.0-q[~excl] #q_head
        #all_r_spontcorr_ck[ai_i] = pp.weighted_corr(spontcorr_,ck_,weights_)
        print("\r",end="")
        
    sorted_neurons = np.argsort(all_r_spontcorr_ck)

    if shuffle:
        shuffled_r = np.zeros(shuffle_n+1)
    for shuffle_i in np.arange(shuffle_n+1):
        if shuffle and shuffle_i>0: np.random.shuffle(sorted_neurons)
        exclnan = np.isnan(all_r_spontcorr_ck[sorted_neurons])
        nonnann = np.sum(~exclnan)
        print("nonnann",nonnann)
        all_top_neurons_r_spontcorr_ck = np.zeros(nonnann)
        for top_n in np.arange(nonnann): #[top_n_opt]:#
            print("shuffle_k_i ",shuffle_k_i," top_n ",top_n,end="")
            top_neurons = sorted_neurons[~exclnan][-top_n:]

            ck_top_neurons = funa.get_correlation_from_kernels_map(km,occ3,js=top_neurons,set_unknown_to_zero=False)#=True)
            ck_top_neurons[q>0.05]=np.nan 
            nanmask = np.isnan(ck_top_neurons)*np.isnan(ck_top_neurons.T)
            ck_top_neurons = 0.5*(np.nansum([ck_top_neurons,ck_top_neurons.T],axis=0))
            ck_top_neurons[nanmask] = np.nan

            excl = np.isnan(spontcorr) | np.isnan(ck_top_neurons) | ondiag
            ##FIXME excl = excl_ | np.isnan(ck_top_neurons)
            spontcorr_ = spontcorr[~excl]
            ck_ = ck_top_neurons[~excl]
            top_neurons_r_spontcorr_ck = np.corrcoef([spontcorr_,ck_])[0,1]
            all_top_neurons_r_spontcorr_ck[top_n] = top_neurons_r_spontcorr_ck
            if shuffle: shuffled_r[shuffle_i] = top_neurons_r_spontcorr_ck
            print("\r",end="")
            
            #print(top_neurons)
            #print([funa.neuron_ids[tn] for tn in top_neurons])
            #print(top_neurons_r_spontcorr_ck)
            if not shuffle and not shuffle_k and top_n==top_n_opt:
                fig = plt.figure(2)
                ax = fig.add_subplot(111)
                ax.plot(spontcorr_,ck_,'o',alpha=0.3)
                ax.axvline(0,c="k",alpha=0.5)
                ax.axhline(0,c="k",alpha=0.5)
                ax.set_xlabel(r"$\langle$spontaneous correlation$\rangle_{ds}$")
                ax.set_ylabel("kernel-derived correlation (top "+str(len(top_neurons))+" neurons)")
                fig.tight_layout()
                if not unc31:
                    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kernel_vs_spont_scatter.png",dpi=300,bbox_inches="tight")
                    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS3/funatlas_vs_correlations_opt_corr_kernel_vs_spont_scatter.pdf",bbox_inches="tight")
                
                fig22 = plt.figure(22)
                ax = fig22.add_subplot(111)
                ax.set_facecolor((0.4,0.4,0.4))
                ctnplot = funa_spont.reduce_to_head(ck_top_neurons)
                np.fill_diagonal(ctnplot,np.nan)
                ax.imshow(ctnplot,interpolation="none")
                fig22.tight_layout()
                if not unc31:
                    fig22.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kernel.png",dpi=300,bbox_inches="tight")
                    fig22.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS3/funatlas_vs_correlations_opt_corr_kernel.pdf",bbox_inches="tight")
                
                fig23 = plt.figure(23)
                ax = fig23.add_subplot(111)
                ax.set_facecolor((0.4,0.4,0.4))
                #ctnplot = funa_spont.reduce_to_head(ck_top_neurons)
                sorter_i = np.loadtxt("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/sorter_i.txt").astype(int)
                sorter_j = np.loadtxt("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/sorter_j.txt").astype(int)
                sorter2 = np.loadtxt("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/sorter_2.txt").astype(int)
                ctnplot = ck_top_neurons[sorter_i][:,sorter_j][sorter2][:,sorter2]
                np.fill_diagonal(ctnplot,np.nan)
                ax.imshow(ctnplot,interpolation="none")
                fig23.tight_layout()
                if not unc31:
                    fig23.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kernel_sorter.png",dpi=300,bbox_inches="tight")
                    fig23.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS3/funatlas_vs_correlations_opt_corr_kernel_sorted.pdf",bbox_inches="tight")
                
    if shuffle_k:
        shuffled_k_r.append(all_top_neurons_r_spontcorr_ck)
        shuffled_k_r_max[shuffle_k_i] = np.max(all_top_neurons_r_spontcorr_ck)
    
print([funa.neuron_ids[tn] for tn in top_neurons]) #head_ids
print("TAKE THIS FOR THE BAR PLOT top_n_opt=",top_n_opt," correlation",(all_top_neurons_r_spontcorr_ck[top_n_opt]),"funatlas_vs_correlation_opt_bar_plot.py")
print("OPT DRIVING NEURONS",[funa_spont.neuron_ids[tn] for tn in sorted_neurons[~exclnan][-top_n_opt:][::-1]])
    
if not shuffle:
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.plot(all_r_spontcorr_ck)
    fig.tight_layout()

    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.plot(all_top_neurons_r_spontcorr_ck)
    ax.set_xlabel("top n")
    ax.set_ylabel("r of kernel-derived correlations\nand spontaneous correlations")
    fig.tight_layout()
    if not unc31:
        fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kernel_vs_spont.png",dpi=300,bbox_inches="tight")
    
if shuffle:
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    ax.hist(shuffled_r[1:],density=True,bins=30,label="shuffled")
    ax.axvline(shuffled_r[0],c="k",label="optimal")
    ax.set_xlabel("correlation coefficient of spontaneous correlations and\nkernels-derived correlations with "+str(top_n_opt)+" driving neurons")
    ax.set_ylabel("density")
    fig.tight_layout()
    #fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kernel_vs_spont_shuffled.png",dpi=300,bbox_inches="tight")
    
if shuffle_k:
    fig = plt.figure(5)
    ax = fig.add_subplot(111)
    print(len(shuffled_k_r))
    ax.plot(shuffled_k_r[0],c="k")
    for skr in shuffled_k_r[1:]:
        ax.plot(skr,lw=1,alpha=0.3,c="C0")
    ax.set_xlabel("top n")
    ax.set_ylabel("r of kernel-derived correlations\nand spontaneous correlations")
    fig.tight_layout()
    ###fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kernel_vs_spont_shuffled_k.png",dpi=300,bbox_inches="tight")
    
    fig = plt.figure(6)
    ax = fig.add_subplot(111)
    ax.hist(shuffled_k_r_max[1:],density=True,bins=30,label="shuffled")
    ax.axvline(shuffled_k_r_max[0],c="k",label="optimal")
    ax.set_xlabel("correlation coefficient of spontaneous correlations and\nkernels-derived correlations with "+str(top_n_opt)+" driving neurons")
    ax.set_ylabel("density")
    fig.tight_layout()
    ###fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations_opt_corr_kernel_vs_spont_shuffled_k.png",dpi=300,bbox_inches="tight")


plt.show()
