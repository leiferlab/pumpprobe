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
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
nan_th = 0.3
save = "--no-save" not in sys.argv
save_cache = "--save-cache" in sys.argv
vmax = 0.4
cmap = "coolwarm"
two_min_occ = "--two-min-occ" in sys.argv
alpha_qvalues = "--alpha-qvalues" in sys.argv
alpha_kolmogorov_smirnov = "--alpha-kolmogorov-smirnov" in sys.argv
alpha_occ1 = not alpha_qvalues and not alpha_kolmogorov_smirnov
alphamax_set = None
alphamin_set = None
alphalbl_set = None
enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv
to_paper = "--to-paper" in sys.argv
plot = "--no-plot" not in sys.argv
figsize = (12,10)
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--nan-th": nan_th = float(sa[1])
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
    if sa[0] == "--figsize": figsize = [int(sb) for sb in sa[1].split(",")]

# To determine which control distribution to use
strain = "unc31" if ds_tags == "unc31" else ""
    
# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,          
                 "photobl_appl":True,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file}

funa = pp.Funatlas.from_datasets(
                ds_list,
                merge_bilateral=merge_bilateral,
                merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                signal_kwargs=signal_kwargs,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck,
                verbose=False)

#funa.export_to_txt("/projects/LEIFER/francesco/funatlas/exported_data/")

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ,req_auto_response=req_auto_response)
# If occ2 needs to be filtered
occ1,occ2 = funa.filter_occ12_from_sysargv(occ2,sys.argv)
                 
occ3 = funa.get_observation_matrix(req_auto_response=req_auto_response)
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
    qvalues = funa.get_kolmogorov_smirnov_q(inclall_occ2,strain=strain)
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
            
            y = funa.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,i]
            nan_mask = funa.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
                                         
            if np.sum(nan_mask)>nan_th*len(y): continue
                                         
            if signal_range is None:    
                pre = np.average(y[:shift_vol])                                 
                if pre == 0: continue
                dy = np.average( y[shift_vol:] - pre )/pre
            else:
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
    gs = fig1.add_gridspec(1,10)
    ax = fig1.add_subplot(gs[0,:9])
    cax = fig1.add_subplot(gs[0,9:])
    #ax.fill_between(np.arange(mappa.shape[1]), 0, y2=mappa.shape[0], hatch='//////', zorder=0, fc='white')
    ax.imshow(0.*np.ones_like(mappa),cmap="Greys",vmax=1,vmin=0)
    blank_mappa = np.copy(mappa)
    blank_mappa[~np.isnan(mappa)] = 0.1
    ax.imshow(blank_mappa,cmap="Greys",vmin=0,vmax=1)
    im = ax.imshow(mappa,cmap=cmap,vmin=-vmax,vmax=vmax,alpha=alphas,interpolation="nearest")
    pp.make_alphacolorbar(cax,-vmax,vmax,0.1,alphamin,alphamax,2,cmap=cmap,around=1,lbl_lg=True)
    cax.set_xlabel(alphalbl,fontsize=15)
    cax.set_ylabel(r'$\langle\Delta F/F\rangle_t$',fontsize=15)
    cax.tick_params(labelsize=15)
    ax.set_xlabel("stimulated",fontsize=30)
    ax.set_ylabel("responding",fontsize=30)
    ax.set_xticks(np.arange(len(sorter_j)))
    ax.set_yticks(np.arange(len(sorter_i)))
    if SIM:
        ax.set_xticklabels(funa.SIM_head_ids[sorter_j],fontsize=5,rotation=90)
        ax.set_yticklabels(funa.SIM_head_ids[sorter_i],fontsize=5)
    else:
        ax.set_xticklabels(funa.head_ids[sorter_j],fontsize=5,rotation=90)
        ax.set_yticklabels(funa.head_ids[sorter_i],fontsize=5)
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
    ax.set_xlim(-0.5,lim)
    if stamp: pp.provstamp(ax,-.1,-.1," ".join(sys.argv))


################################
# COMPARE TO ACTIVITY CONNECTOME
################################
skip_act_conn = True
if not sort and not pop_nans and not skip_act_conn:
    act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome/activity_connectome_bilateral_merged.txt")
    print(act_conn.shape)
    act_conn = funa.reduce_to_head(act_conn)
    dFF = mappa[~(np.isnan(mappa)^np.isinf(mappa))]
    dFF_a = mappa[~(np.isnan(mappa)^np.isinf(mappa))]*alphas[~(np.isnan(mappa)^np.isinf(mappa))]
    act_conn_x = act_conn[~(np.isnan(mappa)^np.isinf(mappa))]
    def fitf(x,a):
        return a*x
        
    p1,_ = curve_fit(fitf,act_conn_x,dFF,p0=[1])
    x1 = np.linspace(np.min(act_conn_x),np.max(act_conn_x),10)
    line1 = fitf(x1,p1)
    p2,_ = curve_fit(fitf,act_conn_x,dFF_a,p0=[1])
    line2 = fitf(x1,p2)

    var1 = np.sum((dFF-np.average(dFF))**2)
    r2_1 = 1-np.sum( (dFF-fitf(act_conn_x,p1))**2 )/var1
    var2 = np.sum((dFF_a-np.average(dFF_a))**2)
    r2_2 = 1-np.sum( (dFF_a-fitf(act_conn_x,p2))**2 )/var2
    print(r2_1,r2_2)
    
    fig2 = plt.figure(2)
    ax=fig2.add_subplot(111)
    ax.plot(np.ravel(act_conn_x),np.ravel(dFF),'o')
    ax.plot(x1,line1)
    ax.set_xlabel("activity connectome")
    ax.set_ylabel("average dF/F")
    ax.set_title("R^2 = 1 - e^2/sigma^2 = "+str(np.around(r2_1,3)))
    ax.set_ylim(-0.6,1.7)
    fig2.tight_layout()
    fig2.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/dFF_vs_act_conn.png",dpi=300,bbox_inches="tight")
    
    fig3 = plt.figure(3)
    ax=fig3.add_subplot(111)
    ax.plot(np.ravel(act_conn_x),np.ravel(dFF_a),'o')
    ax.plot(x1,line2)
    ax.set_xlabel("activity connectome")
    ax.set_ylabel("average dF/F * (1-q)")
    ax.set_title("R^2 = 1 - e^2/sigma^2 = "+str(np.around(r2_2,3)))
    fig3.tight_layout()
    fig3.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/dFF_1-q_vs_act_conn.png",dpi=300,bbox_inches="tight")

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
if save_cache:
    if ds_tags=="unc31":
        add_file_name = "_unc31"
    else:
        add_file_name = ""
    np.savetxt(folder+txt_fname.split(".")[0]+"_cache"+add_file_name+".txt",mappa_full)
    np.savetxt(folder+txt_fname.split(".")[0]+"_cache_occ3"+add_file_name+".txt",occ3_full)
    if alpha_kolmogorov_smirnov:
        np.savetxt(folder+txt_fname.split(".")[0]+"_cache_q"+add_file_name+".txt",qvalues_orig)

plt.show()
