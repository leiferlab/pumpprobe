import numpy as np, matplotlib.pyplot as plt
from scipy.stats import kstest
import pumpprobe as pp

c_in_wt = pp.c_in_wt
c_not_in_wt = pp.c_not_in_wt
c_wt = pp.c_wt
c_unc31 = pp.c_unc31
cy3 = "k"
cy4 = "gray"

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

funa = pp.Funatlas()
bilateral_companions = funa.get_bilateral_companions()
funa.load_aconnectome_from_file(chem_th=0,gap_th=0)

aconn = funa.aconn_chem + funa.aconn_gap
actconn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_no_merge.txt")
actconn_fit_wt = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2_inverted/activity_connectome_no_merge.txt")
actconn_fit_unc31 = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2_inverted_unc31/activity_connectome_no_merge.txt")

intensity_map_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache.txt")
occ3_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_occ3.txt")
q_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q.txt")
tost_q_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_tost_q.txt")
intensity_map_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_unc31.txt")
occ3_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_occ3_unc31.txt")
q_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q_unc31.txt")
tost_q_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_tost_q_unc31.txt")

occ1slowbool = np.loadtxt("/projects/LEIFER/francesco/funatlas/occ1slowbool.txt")

in_wt = q_wt<0.05
not_in_wt = tost_q_wt<0.05
in_unc31 = q_unc31<0.05
not_in_unc31 = tost_q_unc31<0.05
ondiag = np.zeros_like(q_wt,dtype=bool); np.fill_diagonal(ondiag,True)

excl_wt = np.logical_or(np.isnan(q_wt),ondiag)
excl_unc31 = np.logical_or(np.isnan(q_unc31),ondiag)

# Compute the significance of the difference between the distributions of dF/F
# for actconn above and below this threshold (determined empirically looking at
# the plot)
split_distr_th = 0.1

dff1 = intensity_map_wt[np.logical_and(actconn<split_distr_th,~np.isnan(q_wt))]
dff2 = intensity_map_wt[np.logical_and(actconn>=split_distr_th,~np.isnan(q_wt))]
_,p_all = kstest(dff1,dff2,alternative="greater")
print("p of CDF dff1 being greater than CDF dff2 (all)", p_all)
dff1 = intensity_map_wt[np.logical_and(actconn<split_distr_th,in_wt)]
dff2 = intensity_map_wt[np.logical_and(actconn>=split_distr_th,in_wt)]
_,p = kstest(dff1,dff2,alternative="greater")
print("p of CDF dff1 being greater than CDF dff2 (q<0.05)", p)
actconn1 = actconn[in_wt]
actconn2 = actconn[not_in_wt]
_,p = kstest(actconn1,actconn2,alternative="less")
print("p of CDF actconn1 being greater than CDF actconn2", p)

#####################################################
# COMPUTE CORRELATION COEFFICIENTS WITH INTENSITY MAP
#####################################################
# WT
r_fconn_actconn_wt = np.corrcoef(actconn[~excl_wt],intensity_map_wt[~excl_wt])[0,1]
r_fconn_actconn_split_distr_wt = np.corrcoef(actconn[np.logical_and(actconn>split_distr_th,~excl_wt)],intensity_map_wt[np.logical_and(actconn>split_distr_th,~excl_wt)])[0,1]
r_fconn_actconn_fit_wt = np.corrcoef(actconn_fit_wt[~excl_wt],intensity_map_wt[~excl_wt])[0,1]
# Fit with intercept 0
R2_fconn_actconn_wt = pp.R2(actconn[~excl_wt],intensity_map_wt[~excl_wt])
R2_fconn_actconn_split_distr_wt = pp.R2(actconn[np.logical_and(actconn>split_distr_th,~excl_wt)],intensity_map_wt[np.logical_and(actconn>split_distr_th,~excl_wt)])
R2_fconn_actconn_fit_wt = pp.R2(actconn_fit_wt[~excl_wt],intensity_map_wt[~excl_wt])

excl_fastslow_wt = np.logical_or(excl_wt,occ1slowbool)
R2_fconn_actconn_wt_fast = pp.R2(actconn[~excl_fastslow_wt],intensity_map_wt[~excl_fastslow_wt])
R2_fconn_actconn_fit_wt_fast = pp.R2(actconn_fit_wt[~excl_fastslow_wt],intensity_map_wt[~excl_fastslow_wt])


print("### wt")
print("r_fconn_actconn_wt",r_fconn_actconn_wt)
print("r_fconn_actconn_split_distr_wt",r_fconn_actconn_split_distr_wt)
print("r_fconn_actconn_fit_wt",r_fconn_actconn_fit_wt)
print("R2_fconn_actconn_wt",R2_fconn_actconn_wt)
print("R2_fconn_actconn_split_distr_wt",R2_fconn_actconn_split_distr_wt)
print("R2_fconn_actconn_fit_wt",R2_fconn_actconn_fit_wt)
print("R2_fconn_actconn_wt_fast",R2_fconn_actconn_wt_fast)
print("R2_fconn_actconn_fit_wt_fast",R2_fconn_actconn_fit_wt_fast)

# unc31
r_fconn_actconn_unc31 = np.corrcoef(actconn[~excl_unc31],intensity_map_unc31[~excl_unc31])[0,1]
r_fconn_actconn_split_distr_unc31 = np.corrcoef(actconn[np.logical_and(actconn>split_distr_th,~excl_unc31)],intensity_map_unc31[np.logical_and(actconn>split_distr_th,~excl_unc31)])[0,1]
r_fconn_actconn_fit_unc31 = np.corrcoef(actconn_fit_unc31[~excl_unc31],intensity_map_unc31[~excl_unc31])[0,1]
# Fit with intercept 0
R2_fconn_actconn_unc31 = pp.R2(actconn[~excl_unc31],intensity_map_unc31[~excl_unc31])
R2_fconn_actconn_split_distr_unc31 = pp.R2(actconn[np.logical_and(actconn>split_distr_th,~excl_unc31)],intensity_map_unc31[np.logical_and(actconn>split_distr_th,~excl_unc31)])
R2_fconn_actconn_fit_unc31 = pp.R2(actconn_fit_unc31[~excl_unc31],intensity_map_unc31[~excl_unc31])

excl_fastslow_unc31 = np.logical_or(excl_unc31,occ1slowbool)
R2_fconn_actconn_unc31_fast = pp.R2(actconn[~excl_fastslow_unc31],intensity_map_unc31[~excl_fastslow_unc31])
R2_fconn_actconn_fit_unc31_fast = pp.R2(actconn_fit_unc31[~excl_fastslow_unc31],intensity_map_unc31[~excl_fastslow_unc31])

print("### unc31")
print("r_fconn_actconn_unc31",r_fconn_actconn_unc31)
print("r_fconn_actconn_split_distr_unc31",r_fconn_actconn_split_distr_unc31)
print("r_fconn_actconn_fit_unc31",r_fconn_actconn_fit_unc31)
print("R2_fconn_actconn_unc31",R2_fconn_actconn_unc31)
print("R2_fconn_actconn_split_distr_unc31",R2_fconn_actconn_split_distr_unc31)
print("R2_fconn_actconn_fit_unc31",R2_fconn_actconn_fit_unc31)
print("R2_fconn_actconn_unc31_fast",R2_fconn_actconn_unc31_fast)
print("R2_fconn_actconn_fit_unc31_fast",R2_fconn_actconn_fit_unc31_fast)

##############################
# COMPARE BILATERAL COMPANIONS
##############################
fconn_wt_bilateral = np.zeros(len(bilateral_companions))
fconn_unc31_bilateral = np.zeros(len(bilateral_companions))
q_wt_bil_nanmask = np.zeros(len(bilateral_companions),dtype=bool)
q_unc31_bil_nanmask = np.zeros(len(bilateral_companions),dtype=bool)
actconn_bilateral =  np.zeros(len(bilateral_companions))
for j in np.arange(len(bilateral_companions)):
    i = bilateral_companions[j]
    fconn_wt_bilateral[j] = intensity_map_wt[i,j]
    fconn_unc31_bilateral[j] = intensity_map_unc31[i,j]
    q_wt_bil_nanmask[j] = np.isnan(q_wt[i,j])
    q_unc31_bil_nanmask[j] = np.isnan(q_unc31[i,j])
    actconn_bilateral[j] = actconn[i,j]

excl_bilateral_wt = q_wt_bil_nanmask
excl_bilateral_unc31 = q_unc31_bil_nanmask
R2_fconn_actconn_wt_bilateral = pp.R2(fconn_wt_bilateral[~excl_bilateral_wt],actconn_bilateral[~excl_bilateral_wt])
R2_fconn_actconn_unc31_bilateral = pp.R2(fconn_unc31_bilateral[~excl_bilateral_unc31],actconn_bilateral[~excl_bilateral_unc31])

print("### bilateral companions")
print("R2_fconn_actconn_wt_bilateral",R2_fconn_actconn_wt_bilateral)
print("R2_fconn_actconn_unc31_bilateral",R2_fconn_actconn_unc31_bilateral)

################################################################
# ESTIMATE ERRORS IN CORRELATION COEFFICIENTS WITH INTENSITY MAP
################################################################
r = []
r_fit = []
for k in np.arange(actconn.shape[0]):
    excl2 = np.copy(excl_wt)
    excl2[k,:] =  True
    r.append(np.corrcoef(actconn[~excl2],intensity_map_wt[~excl2])[0,1])
    r_fit.append(np.corrcoef(actconn_fit_wt[~excl2],intensity_map_wt[~excl2])[0,1])
r_fconn_actconn_wt_err = np.std(r)
r_fconn_actconn_fit_wt_err = np.std(r_fit)

where = np.where(~excl_unc31)
r = []
r_fit = []
for k in np.arange(actconn.shape[0]):
    excl2 = np.copy(excl_unc31)
    excl2[k,:] =  True
    r.append(np.corrcoef(actconn[~excl2],intensity_map_unc31[~excl2])[0,1])
    r_fit.append(np.corrcoef(actconn_fit_unc31[~excl2],intensity_map_unc31[~excl2])[0,1])
r_fconn_actconn_unc31_err = np.std(r)
r_fconn_actconn_fit_unc31_err = np.std(r_fit)

#####################################################################################
# COMPUTE CORRELATION COEFFICIENTS WITH INTENSITY MAP AS A FUNCTION OF OCC3 THRESHOLD
#####################################################################################
occ3_ths = np.arange(max(np.max(occ3_wt),np.max(occ3_unc31)))
n_pairs_wt_occ3 = np.zeros(len(occ3_ths))
n_pairs_unc31_occ3 = np.zeros(len(occ3_ths))

r_fconn_actconn_wt_occ3 = np.zeros(len(occ3_ths))*np.nan
r_fconn_actconn_fit_wt_occ3 = np.zeros(len(occ3_ths))*np.nan
R2_fconn_actconn_wt_occ3 = np.zeros(len(occ3_ths))*np.nan
R2_fconn_actconn_fit_wt_occ3 = np.zeros(len(occ3_ths))*np.nan
r_fconn_actconn_unc31_occ3 = np.zeros(len(occ3_ths))*np.nan
r_fconn_actconn_fit_unc31_occ3 = np.zeros(len(occ3_ths))*np.nan
R2_fconn_actconn_unc31_occ3 = np.zeros(len(occ3_ths))*np.nan
R2_fconn_actconn_fit_unc31_occ3 = np.zeros(len(occ3_ths))*np.nan

for oi in np.arange(len(occ3_ths)):
    occ3_th = occ3_ths[oi]
    excl_wt_o3 = np.logical_or(excl_wt,occ3_wt<occ3_th)
    excl_unc31_o3 = np.logical_or(excl_unc31,occ3_unc31<occ3_th)
    
    n_pairs_wt_occ3[oi] = np.sum(~excl_wt_o3)
    n_pairs_unc31_occ3[oi] = np.sum(~excl_unc31_o3)
    
    # WT
    r_fconn_actconn_wt_occ3[oi] = np.corrcoef(actconn[~excl_wt_o3],intensity_map_wt[~excl_wt_o3])[0,1]
    r_fconn_actconn_fit_wt_occ3[oi] = np.corrcoef(actconn_fit_wt[~excl_wt_o3],intensity_map_wt[~excl_wt_o3])[0,1]
    # Fit with intercept 0
    try: 
        R2_fconn_actconn_wt_occ3[oi] = pp.R2(actconn[~excl_wt_o3],intensity_map_wt[~excl_wt_o3])
        R2_fconn_actconn_fit_wt_occ3[oi] = pp.R2(actconn_fit_wt[~excl_wt_o3],intensity_map_wt[~excl_wt_o3])
    except: pass

    # unc31
    r_fconn_actconn_unc31_occ3[oi] = np.corrcoef(actconn[~excl_unc31_o3],intensity_map_unc31[~excl_unc31_o3])[0,1]
    r_fconn_actconn_fit_unc31_occ3[oi] = np.corrcoef(actconn_fit_unc31[~excl_unc31_o3],intensity_map_unc31[~excl_unc31_o3])[0,1]
    # Fit with intercept 0
    try: 
        R2_fconn_actconn_unc31_occ3[oi] = pp.R2(actconn[~excl_unc31_o3],intensity_map_unc31[~excl_unc31_o3])
        R2_fconn_actconn_fit_unc31_occ3[oi] = pp.R2(actconn_fit_unc31[~excl_unc31_o3],intensity_map_unc31[~excl_unc31_o3])
    except: pass


#########################################
# COMPUTE CORRELATION COEFFICIENTS WITH Q
#########################################
# WT
r_q_actconn_wt = np.corrcoef(actconn[~excl_wt],1.-q_wt[~excl_wt])[0,1]
r_q_actconn_split_distr_wt = np.corrcoef(actconn[np.logical_and(actconn>split_distr_th,~excl_wt)],1.-q_wt[np.logical_and(actconn>split_distr_th,~excl_wt)])[0,1]
r_q_actconn_fit_wt = np.corrcoef(actconn_fit_wt[~excl_wt],1.-q_wt[~excl_wt])[0,1]
# Fit with intercept 0
R2_q_actconn_wt = pp.R2(actconn[~excl_wt],1.-q_wt[~excl_wt])
R2_q_actconn_split_distr_wt = pp.R2(actconn[np.logical_and(actconn>split_distr_th,~excl_wt)],1.-q_wt[np.logical_and(actconn>split_distr_th,~excl_wt)])
R2_q_actconn_fit_wt = pp.R2(actconn_fit_wt[~excl_wt],1.-q_wt[~excl_wt])

print("### wt")
print("r_q_actconn_wt",r_q_actconn_wt)
print("r_q_actconn_split_distr_wt",r_q_actconn_split_distr_wt)
print("r_q_actconn_fit_wt",r_q_actconn_fit_wt)
print("R2_q_actconn_wt",R2_q_actconn_wt)
print("R2_q_actconn_split_distr_wt",R2_q_actconn_split_distr_wt)
print("R2_q_actconn_fit_wt",R2_q_actconn_fit_wt)

# unc31
r_q_actconn_unc31 = np.corrcoef(actconn[~excl_unc31],1.-q_unc31[~excl_unc31])[0,1]
r_q_actconn_split_distr_unc31 = np.corrcoef(actconn[np.logical_and(actconn>split_distr_th,~excl_unc31)],1.-q_unc31[np.logical_and(actconn>split_distr_th,~excl_unc31)])[0,1]
r_q_actconn_fit_unc31 = np.corrcoef(actconn_fit_unc31[~excl_unc31],1.-q_unc31[~excl_unc31])[0,1]
# Fit with intercept 0
R2_q_actconn_unc31 = pp.R2(actconn[~excl_unc31],1.-q_unc31[~excl_unc31])
R2_q_actconn_split_distr_unc31 = pp.R2(actconn[np.logical_and(actconn>split_distr_th,~excl_unc31)],1.-q_unc31[np.logical_and(actconn>split_distr_th,~excl_unc31)])
R2_q_actconn_fit_unc31 = pp.R2(actconn_fit_unc31[~excl_unc31],1.-q_unc31[~excl_unc31])

print("### unc31")
print("r_q_actconn_unc31",r_q_actconn_unc31)
print("r_q_actconn_split_distr_unc31",r_q_actconn_split_distr_unc31)
print("r_q_actconn_fit_unc31",r_q_actconn_fit_unc31)
print("R2_q_actconn_unc31",R2_q_actconn_unc31)
print("R2_q_actconn_split_distr_unc31",R2_q_actconn_split_distr_unc31)
print("R2_q_actconn_fit_unc31",R2_q_actconn_fit_unc31)

##################################
# MAIN FIGURES
##################################
##################################
# WT
##################################
fig = plt.figure(2)
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

x1,y1 = actconn[in_wt],intensity_map_wt[in_wt]
x2,y2 = actconn[not_in_wt],intensity_map_wt[not_in_wt]
binsx = np.logspace(-11,np.log10(np.max(x1)),30)
binw=0.02;y1y2max=max(np.max(np.abs(y1)),np.max(np.abs(y2)));lim=(int(y1y2max/binw)+1)*binw
binsy = np.arange(-lim, lim + binw, binw)
pp.scatter_hist(x1, y1, ax, ax_histx, ax_histy, label="functionally connected",binsx=binsx,binsy=binsy,alpha_scatter=0.5,alpha_hist=0.5,color=c_in_wt)
pp.scatter_hist(x2, y2, ax, ax_histx, ax_histy, label="functionally not connected",binsx=binsx,binsy=binsy,alpha_scatter=0.5,alpha_hist=0.5,color=c_not_in_wt)

y3 = intensity_map_wt[np.logical_and(in_wt,actconn>split_distr_th)]
y4 = intensity_map_wt[np.logical_and(in_wt,actconn<=split_distr_th)]
ax_histy.hist(y3,bins=binsy,orientation='horizontal',alpha=0.5,histtype="step",color=cy3)
ax_histy.hist(y4,bins=binsy,orientation='horizontal',alpha=0.5,histtype="step",color=cy4)
ax_histy.plot((0,70,70,0),(1,1,-0.5,-0.5),c="k",alpha=0.5)
ax_histy.text(90,1,"panel c",color="k",alpha=0.5)

ax.axvline(split_distr_th,color="k",alpha=0.3)
ax.set_xscale("log")
ax_histx.set_xscale("log")
ax.set_xlabel("Anatomy-derived response $\\Delta V$ (V)\nbiophysical model")
ax.set_ylabel(r"$\Delta F/F$")
ax.legend()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_wt.pdf",dpi=300,bbox_inches="tight")

# Zoomed-in inset for the right marginal distribution
fig = plt.figure(21,figsize=(2,4))
ax = fig.add_subplot(111)
ax.hist(y1, bins=binsy, orientation='horizontal',alpha=0.5,label="functionally conn.",color=c_in_wt)
ax.hist(y2, bins=binsy, orientation='horizontal',alpha=0.5,label="functionally not conn.",color=c_not_in_wt)
ax.hist(y3,bins=binsy,orientation='horizontal',alpha=0.5,histtype="step",lw=2,label="functionally and anatomically conn.",color=cy3)
ax.hist(y4,bins=binsy,orientation='horizontal',alpha=0.5,histtype="step",lw=2,label="functionally conn. but anatomically not conn.",color=cy4)
ax.set_ylim(-0.5,1.0)
ax.set_xlim(0,70)
ax.set_yticks([-0.5,0.,0.5,1.0])
ax.set_yticklabels(["-0.5","0","0.5","1.0"])
ax.set_ylabel(r"$\Delta F/F$")
ax.set_xlabel("number of pairs")
ax.legend(bbox_to_anchor=(1,1), loc="upper left")
ax.text(80,0.1,"**** Functionally $and$ anatomically connected\npairs have $\\Delta F/F$ larger than functionally\nconnected but anatomically $not$ connected pairs.\n(p<10$^{-36}$ one-sided KS test)",fontsize=10)
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_wt_inset.pdf",dpi=300,bbox_inches="tight")

# Violin
y3b = intensity_map_wt[np.logical_and(~np.isnan(q_wt),actconn>split_distr_th)]
y4b = intensity_map_wt[np.logical_and(~np.isnan(q_wt),actconn<=split_distr_th)]
_,p_all = kstest(y3b,y4b,alternative="less")
fig = plt.figure(22,figsize=(4,8))
ax = fig.add_subplot(111)
ax.violinplot([y4b,y3b],points=1000)
maxheight = 1.05*max(np.max(y3b),np.max(y4b))
ax.plot((1,2),(maxheight,maxheight),c="k")
ax.text(1.5,maxheight,pp.p_to_stars(p_all),ha="center")
ax.set_xticks([1,2])
ax.set_xticklabels([r"$\Delta V\leq"+str(split_distr_th)+"$",r"$\Delta V>"+str(split_distr_th)+"$",],fontsize=22)
ax.set_yticks([0,1,2])
ax.set_yticklabels(["0","1","2"],fontsize=30)
ax.set_ylabel(r"$\Delta F/F$",fontsize=30)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_wt_violin.pdf",dpi=300,bbox_inches="tight")

##################################
# unc31
##################################
fig = plt.figure(3)
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

x1,y1 = actconn[in_unc31],intensity_map_unc31[in_unc31]
x2,y2 = actconn[not_in_unc31],intensity_map_unc31[not_in_unc31]
binsx = np.logspace(-11,np.log10(np.max(x1)),30)
binw=0.02;y1y2max=max(np.max(np.abs(y1)),np.max(np.abs(y2)));lim=(int(y1y2max/binw)+1)*binw
binsy = np.arange(-lim, lim + binw, binw)
pp.scatter_hist(x1, y1, ax, ax_histx, ax_histy, label="connected",binsx=binsx,binsy=binsy,binwidth=0.02,alpha_scatter=0.5,alpha_hist=0.5)
pp.scatter_hist(x2, y2, ax, ax_histx, ax_histy, label="not connected",binsx=binsx,binsy=binsy,binwidth=0.02,alpha_scatter=0.5,alpha_hist=0.5)

y3 = intensity_map_unc31[np.logical_and(in_unc31,actconn>split_distr_th)]
y4 = intensity_map_unc31[np.logical_and(in_unc31,actconn<=split_distr_th)]
ax_histy.hist(y3,bins=binsy,orientation='horizontal',alpha=0.5,histtype="step")
ax_histy.hist(y4,bins=binsy,orientation='horizontal',alpha=0.5,histtype="step")

ax.axvline(split_distr_th,color="k",alpha=0.3)
ax.set_xscale("log")
ax_histx.set_xscale("log")
ax.set_xlabel("Anatomy-derived response (V)\nbiophysical model")
ax.set_ylabel(r"$\Delta F/F$")
ax.legend()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_unc31.pdf",dpi=300,bbox_inches="tight")


##################################
# BAR PLOT r
##################################
fig = plt.figure(4)
ax = fig.add_subplot(111)
bars = [r_fconn_actconn_wt,
        r_fconn_actconn_fit_wt,
        r_fconn_actconn_unc31,
        r_fconn_actconn_fit_unc31
        ]
errs = [r_fconn_actconn_wt_err,
        r_fconn_actconn_fit_wt_err,
        r_fconn_actconn_unc31_err,
        r_fconn_actconn_fit_unc31_err
        ]
print(errs)
x = np.arange(len(bars))/2
ax.bar(x,bars,yerr=errs,width=0.4,align="center")
ax.set_xticks(x)
ax.set_xticklabels(["Head\nWT",
                    "Head\nWT (fit)",
                    "Head\nunc-31",
                    "Head\nunc-31 (fit)",
                    ])
ax.set_ylabel("Pearson's correlation coefficient")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_barplot.pdf",dpi=300,bbox_inches="tight")

##################################
# BAR PLOT R2
##################################
fig = plt.figure(5)
ax = fig.add_subplot(111)
bars = [R2_fconn_actconn_wt,
        R2_fconn_actconn_fit_wt,
        R2_fconn_actconn_unc31,
        R2_fconn_actconn_fit_unc31,
        ]
x = np.arange(len(bars))/2
ax.bar(x,bars,width=0.4,align="center")
ax.set_xticks(x)
ax.set_xticklabels(["Head\nWT",
                    "Head\nWT (fit)",
                    "Head\nunc-31",
                    "Head\nunc-31 (fit)",                    
                    ])
ax.set_ylabel(r"$R^2$")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_barplot_R2.pdf",dpi=300,bbox_inches="tight")

##################################
# LINE PLOT r
##################################
fig = plt.figure(104)
ax = fig.add_subplot(111)
y1 = [r_fconn_actconn_wt,r_fconn_actconn_fit_wt,]
y2 = [r_fconn_actconn_unc31,r_fconn_actconn_fit_unc31]
x = np.arange(2)
ax.plot(x,y1,ls="-",marker="o",label="WT")
ax.plot(x,y2,ls="-",marker="o",label="unc-31")
ax.set_xticks(x)
ax.set_xticklabels(["Head","Head(fit)",])
ax.set_ylabel("Pearson's correlation coefficient")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_lineplot.pdf",dpi=300,bbox_inches="tight")

##################################
# LINE PLOT R2
##################################
fig = plt.figure(105,figsize=(6,7))
ax = fig.add_subplot(111)
y1 = [R2_fconn_actconn_wt,
      R2_fconn_actconn_fit_wt,
      ]
y2 = [R2_fconn_actconn_unc31,
      R2_fconn_actconn_fit_unc31,
      ]
x1 = np.arange(len(y1))
x2 = x1
ax.plot(x2,y2,ls="-",marker="o",label="unc-31", color=pp.c_unc31,lw=3)
ax.plot(x1,y1,ls="-",marker="o",label="WT",color=pp.c_wt,lw=3)
ax.set_xticks(x1)
ax.set_xticklabels(["Anatomical\nweights",
                    "Fitted\nweights",
                    ],fontsize=30)
ax.set_ylim(None, 0.0)
ax.set_yticks([-0.08,-0.04,0.0])
ax.set_yticklabels(["-0.08","-0.04","0.0"],fontsize=30)
ax.set_ylabel("Agrement to anatomy\n$\\Delta$F/F vs $\\Delta$V",fontsize=30)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.legend(fontsize=20)
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_lineplot_R2.pdf",dpi=300,bbox_inches="tight")

#############################################
# PEARSON'S r AS A FUNCTION OF OCC3 THRESHOLD
#############################################
fig = plt.figure(6)
ax = fig.add_subplot(111)
axb = ax.twinx()
ax.plot(occ3_ths[:-1],r_fconn_actconn_wt_occ3[:-1],label="WT",c="C0",ls="-")
ax.plot(occ3_ths[:-1],r_fconn_actconn_fit_wt_occ3[:-1],label="WT (fit)",c="C0",ls="--")
ax.plot(occ3_ths[:-1],r_fconn_actconn_unc31_occ3[:-1],label="unc-31",c="C1",ls="-")
ax.plot(occ3_ths[:-1],r_fconn_actconn_fit_unc31_occ3[:-1],label="unc-31 (fit)",c="C1",ls="--")
axb.plot(occ3_ths[2:-1],n_pairs_wt_occ3[2:-1],c="C0",ls=":",marker="p")
axb.plot(occ3_ths[2:-1],n_pairs_unc31_occ3[2:-1],c="C1",ls=":",marker="p")
ax.set_xlabel("Threshold on number of observations")
ax.set_ylabel("Pearson's correlation coefficient")
axb.set_ylabel("Number of pairs above threshold")
axb.set_ylim(-10,200)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_occ3th_r.pdf",dpi=300,bbox_inches="tight")

####################################
# R2 AS A FUNCTION OF OCC3 THRESHOLD
####################################
fig = plt.figure(7)
ax = fig.add_subplot(111)
axb = ax.twinx()
ax.plot(occ3_ths[:-1],R2_fconn_actconn_wt_occ3[:-1],label="WT",c="C0",ls="-")
ax.plot(occ3_ths[:-1],R2_fconn_actconn_fit_wt_occ3[:-1],label="WT (fit)",c="C0",ls="--")
ax.plot(occ3_ths[:-1],R2_fconn_actconn_unc31_occ3[:-1],label="unc-31",c="C1",ls="-")
ax.plot(occ3_ths[:-1],R2_fconn_actconn_fit_unc31_occ3[:-1],label="unc-31 (fit)",c="C1",ls="--")
axb.plot(occ3_ths[2:-1],n_pairs_wt_occ3[2:-1],c="C0",ls=":",marker="p")
axb.plot(occ3_ths[2:-1],n_pairs_unc31_occ3[2:-1],c="C1",ls=":",marker="p")
ax.set_xlabel("Threshold on number of observations")
ax.set_ylabel(r"$R^2$")
axb.set_ylabel("Number of pairs above threshold")
axb.set_ylim(-10,200)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.legend()
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_anatomy_occ3th_R2.pdf",dpi=300,bbox_inches="tight")

plt.show()

quit()

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
# OLD PLOTS
fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.scatter(aconn[in_wt],intensity_map_wt[in_wt],label="connected")
ax.scatter(aconn[not_in_wt],intensity_map_wt[not_in_wt],label="not connected")
ax.set_xscale("log")
ax.set_xlabel("Number of anatomical contacts")
ax.set_ylabel(r"$\Delta F/F$")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend()
fig.tight_layout()

