import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit

sort = "--sort" in sys.argv
pop_nans = "--pop-nans" in sys.argv
stamp = "--no-stamp" not in sys.argv
SIM = "--sim" in sys.argv

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "mutant"
ds_exclude_i = []
inclall_occ = "--inclall-occ" in sys.argv
signal_range = None
smooth_mode = "sg_causal"
smooth_n = 13
smooth_poly = 1
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
nan_th = 0.3
correct_decaying = "--correct-decaying" in sys.argv
if not correct_decaying: print("NOT USING THE NEW CORRECTION OF DECAYING RESPONSES")
save = "--no-save" not in sys.argv
save_cache = "--save-cache" in sys.argv
export_to_txt = "--export-to-txt" in sys.argv
vmax = 0.4
cmap = "coolwarm"
two_min_occ = "--two-min-occ" in sys.argv
no_alpha = "--no-alpha" in sys.argv
alpha_qvalues = "--alpha-qvalues" in sys.argv
alpha_kolmogorov_smirnov = "--alpha-kolmogorov-smirnov" in sys.argv
alpha_occ1 = not alpha_qvalues and not alpha_kolmogorov_smirnov and not no_alpha
alphamax_set = None
alphamin_set = None
alphalbl_set = None
ticklabelsize = 5
enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv
to_paper = "--to-paper" in sys.argv
plot = "--no-plot" not in sys.argv
dst = None
figsize = (12,10)
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--nan-th": nan_th = float(sa[1])
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    if sa[0] == "--ds-exclude-tags": 
        ds_exclude_tags=sa[1]
        if ds_exclude_tags == "None": ds_exclude_tags=None
    if sa[0] == "--ds-tags": ds_tags=sa[1]
    if sa[0] == "--ds-exclude-i": ds_exclude_i = [int(sb) for sb in sa[1].split(",")]
    if sa[0] == "--signal-range":
        sb = sa[1].split(",")
        signal_range = [int(sbi) for sbi in sb]
    if sa[0] == "--vmax": vmax = float(sa[1])
    if sa[0] == "--cmap": cmap = sa[1]
    if sa[0] == "--alpha-max": alphamax_set = float(sa[1])
    if sa[0] == "--alpha-min": alphamin_set = float(sa[1])
    if sa[0] == "--alpha-lbl": alphalbl_set = sa[1]
    if sa[0] == "--ticklabelsize": ticklabelsize = int(sa[1])
    if sa[0] == "--figsize": figsize = [int(sb) for sb in sa[1].split(",")]
    if sa[0] == "--dst": dst = sa[1]

print("nan_th",nan_th)
print("ds_exclude_tags",ds_exclude_tags)

# To determine which control distribution to use
strain = "unc31" if ds_tags == "unc31" else ""
    
# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,          
                 "photobl_appl":True,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

funa = pp.Funatlas.from_datasets(
                ds_list,
                merge_bilateral=merge_bilateral,
                merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                signal_kwargs=signal_kwargs,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck,
                verbose=False)

print(len(funa.ds_list))

if export_to_txt:
    if ds_tags=="unc31":
        funa.export_to_txt("/projects/LEIFER/francesco/funatlas/exported_data_unc31/")
    elif ds_tags is None and ds_exclude_tags=="mutant":
        funa.export_to_txt("/projects/LEIFER/francesco/funatlas/exported_data/")
    quit()

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ,req_auto_response=req_auto_response)
# If occ2 needs to be filtered
#occ1,occ2 = funa.filter_occ12_from_sysargv(occ2,sys.argv)
                 
occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=req_auto_response)
occ3_full = np.copy(occ3)

# Setting the transparency of the imshows
if alpha_occ1:
    occ1_head = funa.reduce_to_head(occ1)
    occ1_head = occ1_head/np.max(occ1_head)
    alphalbl = "n responses\n(norm to max)"
    alphamin,alphamax = 0.,0.1#0.25, 1
    if alphamax_set is not None: alphamax = alphamax_set
    if alphamin_set is not None: alphamin = alphamin_set
    alphas = np.clip((occ1_head-alphamin)/(alphamax-alphamin),0,1)
elif alpha_qvalues:
    #qvalues = funa.get_qvalues(occ1,occ3,False)
    qvalues = funa.get_qvalues2(merge_bilateral,req_auto_response)
    print("USING get_qvalues2")
    if SIM:
        qvalues_head = funa.reduce_to_SIM_head(qvalues)
    else:
        qvalues_head = funa.reduce_to_head(qvalues)
    #print(np.sum( (qvalues_head<0.2))/np.prod(qvalues_head.shape))
    qvalues_head[np.isnan(qvalues_head)] = 1.
    alphas = (1.-qvalues_head)
    alphalbl = "1-q"
    alphamin,alphamax = 0.,0.9#0.75
    if alphamax_set is not None: alphamax = alphamax_set
    if alphamin_set is not None: alphamin = alphamin_set
    alphas = np.clip((alphas-alphamin)/(alphamax-alphamin),0,1)
elif alpha_kolmogorov_smirnov:
    if inclall_occ: inclall_occ2 = occ2
    tost_th = 1.2
    qvalues, tost_q = funa.get_kolmogorov_smirnov_q(
                            inclall_occ2,strain=strain,
                            return_tost=True,tost_th=tost_th)
    qvalues_orig = np.copy(qvalues)
    print("Got qvalues")
    qvalues[np.isnan(qvalues)] = 1.
    if SIM:
        qvalues_head = funa.reduce_to_SIM_head(qvalues)
    else:
        qvalues_head = funa.reduce_to_head(qvalues)
    alphas = (1.-qvalues_head)
    alphalbl = "1-q"
    if alphalbl_set is not None: alphalbl = alphalbl_set
    alphamin,alphamax = 0.,1.0
    if alphamax_set is not None: alphamax = alphamax_set
    if alphamin_set is not None: alphamin = alphamin_set
    alphas = np.clip((alphas-alphamin)/(alphamax-alphamin),0,1)
elif no_alpha:
    alphas = np.ones(occ1.shape,dtype=float)

mappa = np.zeros((funa.n_neurons,funa.n_neurons))*np.nan
count = np.zeros((funa.n_neurons,funa.n_neurons))

for ai in np.arange(funa.n_neurons):
    for aj in np.arange(funa.n_neurons):
        if np.isnan(mappa[ai,aj]) and occ3[ai,aj]>0:
            mappa[ai,aj] = 0.0
        
        if two_min_occ:
            if occ1[ai,aj]<2: continue
        
        ys = []
        times = []
        confidences = []

        for occ in occ2[ai,aj]:
            ds = occ["ds"]
            if ds in ds_exclude_i: continue
            ie = occ["stim"]
            i = occ["resp_neu_i"]
            
            # Build the time axis
            i0 = funa.fconn[ds].i0s[ie]
            i1 = funa.fconn[ds].i1s[ie]
            shift_vol = funa.fconn[ds].shift_vol
            Dt = funa.fconn[ds].Dt            
            
            y = funa.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,i]
            nan_mask = funa.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
                                         
            #if np.sum(nan_mask)>nan_th*len(y): continue
            #if not pp.Fconn.nan_ok(nan_mask,nan_th*len(y)): continue
                                         
            if signal_range is None:
                pre = np.average(y[:shift_vol])                                 
                if pre == 0: continue
                
                if correct_decaying:
                    _,_,_,_,_,df_s_unnorm = funa.get_significance_features(
                                funa.sig[ds],i,i0,i1,shift_vol,
                                Dt,nan_th,return_traces=True)
                    if df_s_unnorm is None: continue
                    dy = np.average(df_s_unnorm)/pre
                else:
                    dy = np.average( y[shift_vol:] - pre )/pre
            else:
                print("Not using corrected y")
                #std = np.std(y[:shift_vol-signal_range[0]])
                pre = np.average(y[:shift_vol])                                 
                #dy = np.average(y[shift_vol-signal_range[0]:shift_vol+signal_range[1]+1] - pre)
                dy = np.average(np.abs(y[shift_vol-signal_range[0]:shift_vol+signal_range[1]+1] - pre))
                dy /= pre
            
            if np.isnan(mappa[ai,aj]):
                mappa[ai,aj] = dy
            else:
                mappa[ai,aj] += dy
            count[ai,aj] += 1
            
mappa[count>0] /= count[count>0]

mappa_full = np.copy(mappa)
if SIM:
    mappa = funa.reduce_to_SIM_head(mappa)
else:
    mappa = funa.reduce_to_head(mappa)
if sort:
    mappa,sorter,lim = funa.sort_matrix_nans(mappa,axis=-1,return_all=True)
    sorter_i = sorter_j = sorter
    alphas = alphas[sorter][:,sorter]
elif pop_nans:
    if SIM:
        mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans_SIM(mappa)
        alphas = alphas[sorter_i][:,sorter_j]
    else:
        mappa,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(mappa,return_all=True)
        alphas = alphas[sorter_i][:,sorter_j]
else:
    sorter_i = sorter_j = np.arange(mappa.shape[-1])
    lim = None

if plot:    
    fig1 = plt.figure(1,figsize=figsize)
    if no_alpha:
        ax = fig1.add_subplot(111)
    else:
        gs = fig1.add_gridspec(1,10)
        ax = fig1.add_subplot(gs[0,:9])
        cax = fig1.add_subplot(gs[0,9:])
    #ax.fill_between(np.arange(mappa.shape[1]), 0, y2=mappa.shape[0], hatch='//////', zorder=0, fc='white')
    ax.imshow(0.*np.ones_like(mappa),cmap="Greys",vmax=1,vmin=0)
    blank_mappa = np.copy(mappa)
    blank_mappa[~np.isnan(mappa)] = 0.1
    ax.imshow(blank_mappa,cmap="Greys",vmin=0,vmax=1)
    im = ax.imshow(mappa,cmap=cmap,vmin=-vmax,vmax=vmax,alpha=alphas,interpolation="nearest")
    diagonal = np.diag(np.diag(np.ones_like(mappa)))
    new_diagonal = np.zeros_like(mappa)
    new_diagonal[np.where(diagonal == 1)] = 1
    ax.imshow(new_diagonal, cmap="binary", vmin=0, vmax=1, alpha=new_diagonal, interpolation="nearest")
    if no_alpha:
        plt.colorbar(im, label = r'$\langle\Delta F/F\rangle_t$')
    else:
        pp.make_alphacolorbar(cax,-vmax,vmax,0.1,alphamin,alphamax,2,cmap=cmap,around=1,lbl_lg=True)
        cax.set_xlabel(alphalbl,fontsize=15)
        cax.set_ylabel(r'$\langle\Delta F/F\rangle_t$',fontsize=15)
        cax.tick_params(labelsize=15)
    ax.set_xlabel("stimulated",fontsize=30)
    ax.set_ylabel("responding",fontsize=30)
    ax.set_xticks(np.arange(len(sorter_j)))
    ax.set_yticks(np.arange(len(sorter_i)))
    if SIM:
        ax.set_xticklabels(funa.SIM_head_ids[sorter_j],fontsize=ticklabelsize,rotation=90) # ticklabelsize was 6
        ax.set_yticklabels(funa.SIM_head_ids[sorter_i],fontsize=ticklabelsize)
    else:
        ax.set_xticklabels(funa.head_ids[sorter_j],fontsize=ticklabelsize,rotation=90)
        ax.set_yticklabels(funa.head_ids[sorter_i],fontsize=ticklabelsize)
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
    ax.set_xlim(-0.5,lim+0.5)
    if stamp: pp.provstamp(ax,-.1,-.1," ".join(sys.argv))


if plot: plt.tight_layout()
folder='/projects/LEIFER/francesco/funatlas/'
txt_fname = "funatlas_intensity_map.txt"
if save:
    #folder='/home/leifer/Desktop/fig_transitory/'
    if plot:
        plt.savefig(folder+"funatlas_intensity_map.png", dpi=300, bbox_inches="tight",metadata={"Comment":" ".join(sys.argv)})
        plt.savefig(folder+"funatlas_intensity_map.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(folder+"funatlas_intensity_map.svg", bbox_inches="tight")
    if len(ds_exclude_i)>0: 
        txt_fname = "funatlas_intensity_map_excl_"+"-".join([str(a) for a in ds_exclude_i])+".txt"
    np.savetxt(folder+txt_fname,mappa_full)
    if to_paper and plot:
        plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig2/funatlas_intensity_map.pdf", dpi=600, bbox_inches="tight")
if dst is not None:
    print(dst[-4:])
    if dst[-4:] not in [".pdf",".png"]:
        plt.savefig(dst+"funatlas_intensity_map.pdf", dpi=600, bbox_inches="tight")
        f = open(dst+"funatlas_intensity_map_command_used.txt","w")
    else:
        plt.savefig(dst, dpi=600, bbox_inches="tight")
        f = open(dst+"command_used.txt","w")
    f.write(" ".join(sys.argv))
    f.close()
    
    f = open(dst+"funatlas_intensity_map_for_figure_csv.txt","w")
    if SIM:
        ll_ = funa.SIM_head_ids
    else:
        ll_ = funa.head_ids
    f.write(","+",".join(ll_[sorter_j]))
    s = ""
    for i in np.arange(mappa.shape[0]):
        s += ll_[sorter_i][i]+","+",".join(mappa[i].astype(str))
    s = s[:-1]
    f.write(s)
    
if save_cache:
    if ds_tags=="unc31":
        add_file_name = "_unc31"
    else:
        add_file_name = ""
    np.savetxt(folder+txt_fname.split(".")[0]+"_cache"+add_file_name+".txt",mappa_full)
    np.savetxt(folder+txt_fname.split(".")[0]+"_cache_occ3"+add_file_name+".txt",occ3_full)
    if alpha_kolmogorov_smirnov:
        np.savetxt(folder+txt_fname.split(".")[0]+"_cache_q"+add_file_name+".txt",qvalues_orig)
        np.savetxt(folder+txt_fname.split(".")[0]+"_cache_tost_q"+add_file_name+".txt",tost_q,header="#tost_th: "+str(tost_th))

plt.show()
