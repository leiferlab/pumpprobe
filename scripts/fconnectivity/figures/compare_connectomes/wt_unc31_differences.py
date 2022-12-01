import numpy as np, matplotlib.pyplot as plt
from scipy.stats import kstest
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

c_in_wt = pp.c_in_wt
c_not_in_wt = pp.c_not_in_wt
c_esyn = "C5"
cy3 = "k"
cy4 = "gray"

funa = pp.Funatlas()
funa.load_aconnectome_from_file(chem_th=0,gap_th=0)
aconn = funa.aconn_chem + funa.aconn_gap
aconn = funa.reduce_to_head(aconn)
bilateral_companions = funa.get_bilateral_companions() 

actconn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_no_merge.txt")
np.fill_diagonal(actconn,0)
# Manually select a threshold for the anatomy-derived responses. This
# is the threshold that separates the two distributions that can be
# seen in funatlas_vs_anatomy.py
split_distr_th = 0.1# 3e-4

intensity_map_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache.txt")
q_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q.txt")
tost_q_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_tost_q.txt")
intensity_map_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_unc31.txt")
q_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q_unc31.txt")
tost_q_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_tost_q_unc31.txt")
occ3_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_occ3.txt")

avg_rise_times = np.loadtxt("/projects/LEIFER/francesco/funatlas/avg_rise_times.txt")
avg_rise_times = funa.reduce_to_head(avg_rise_times)

ondiag = np.zeros_like(q_wt,dtype=bool); np.fill_diagonal(ondiag,True)
ondiag = funa.reduce_to_head(ondiag)

#CREATE THE MATRICES OF PAIRS IN OR NOT IN TEH TWO STRAINS BASED ON Q AND TOST-Q
in_wt = q_wt<0.05
not_in_wt = np.logical_and(tost_q_wt<0.05,q_wt>0.2) 
in_unc31 = q_unc31<0.05
# NOT IN UNC-31 IS COMPUTED BELOW USING AN OPTIMIZATION PROCEDURE
#not_in_unc31 = np.logical_and(tost_q_unc31<0.26,q_unc31>0.2)#0.2
#tost_q_wt<0.4,q_wt>0.3 with tost_q_unc31<0.3 gives 2-fold enrichment 1 hop 0 th

intensity_map_wt = funa.reduce_to_head(intensity_map_wt)
intensity_map_unc31 = funa.reduce_to_head(intensity_map_unc31)
occ3_wt = funa.reduce_to_head(occ3_wt)
actconn = funa.reduce_to_head(actconn)

##############################################################
# WT CONNECTIONS THAT GO AWAY IN UNC-31 MUTANT 
# -> ESTIMATE OF EXTRASYNAPTIC CONNECTIONS
##############################################################
##############################################################
# USE THE STRIC TOST Q THRESHOLD FOR NOT IN UNC-31
##############################################################
tost_q_unc31_th = 0.05
not_in_unc31 = np.logical_and(tost_q_unc31<tost_q_unc31_th,q_unc31>0.05)

esyn_strict = np.logical_and(in_wt, not_in_unc31)
esyn_strict[np.isnan(q_wt)] = False
esyn_strict[np.isnan(q_unc31)] = False
esyn_strict = funa.reduce_to_head(esyn_strict)


##############################################################
# COMPUTE NUMBER OF PAIRS IN ESY AS A FUNCTION OF not_in_unc31 
# FALSE-DISCOVERY RATE, AS WELL AS THE Q-VALUES FOR THEIR
# ACTCONN DISTRIBUTION BEING DIFFERENT FROM THE GENERAL ONE.
##############################################################
tost_q_th = np.linspace(0.05,0.4,100)
n_esyn = np.zeros_like(tost_q_th)
p_actconn_actconn_esyn_ = np.zeros_like(tost_q_th)
dist1 = actconn[np.logical_and(funa.reduce_to_head(in_wt),~ondiag)]
for ith in np.arange(len(tost_q_th)):
    tqth = tost_q_th[ith]
    not_in_unc31_ = np.logical_and(tost_q_unc31<tqth,q_unc31>0.05)
    esyn_ = np.logical_and(in_wt, not_in_unc31_)
    esyn_[np.isnan(q_wt)] = False
    esyn_[np.isnan(q_unc31)] = False
    esyn_ = funa.reduce_to_head(esyn_)
    
    n_esyn[ith] = np.sum(esyn_)
    
    dist2 = actconn[np.logical_and(esyn_,~ondiag)]
    _, p_actconn_actconn_esyn_[ith] = kstest(dist1,dist2,alternative="less")

##############################################################
# USE THE OPTIMAL TOST Q THRESHOLD FOR NOT IN UNC-31
##############################################################
tost_q_unc31_th = tost_q_th[np.argmin(p_actconn_actconn_esyn_)]
not_in_unc31 = np.logical_and(tost_q_unc31<tost_q_unc31_th,q_unc31>0.05)

esyn = np.logical_and(in_wt, not_in_unc31)
esyn[np.isnan(q_wt)] = False
esyn[np.isnan(q_unc31)] = False
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS_esyn/candidate_purely_extrasynaptic_connections.txt",esyn)
esyn = funa.reduce_to_head(esyn)
#don't show diagonal/autoresponses
#np.fill_diagonal(esyn,False)
    
#estimated extrasynaptic connections that appear, however, in the connectome
esyn_but_in_aconn = np.logical_and(esyn, aconn!=0)
fconn_in_aconn = np.logical_and(funa.reduce_to_head(in_wt),aconn!=0).astype(int)

#new in unc-31
new_in_unc31 = np.logical_and(not_in_wt, in_unc31)
new_in_unc31[np.isnan(q_wt)] = False
new_in_unc31[np.isnan(q_unc31)] = False
new_in_unc31 = funa.reduce_to_head(new_in_unc31)
#don't show diagonal/autoresponses
#np.fill_diagonal(new_in_unc31,False)
new_in_unc31_in_aconn = np.logical_and(new_in_unc31,aconn!=0)
new_in_unc31_in_aconn = np.logical_and(new_in_unc31,aconn!=0)

###########################################################################
# BUILD DISTRIBUTION OF ACTCONN FOR BILATERAL COMPANIONS, AND COMPARE IT TO 
# AVDL AVDR
###########################################################################
actconn_bilateral = np.zeros(len(bilateral_companions))*np.nan
for j in np.arange(len(bilateral_companions)):
    i = bilateral_companions[j]
    if i==-1: continue
    aj,ai = funa.ai_to_head([j,i])
    actconn_bilateral[j] = actconn[ai,aj]

AVDL_h,AVDR_h = funa.ai_to_head(funa.ids_to_i(["AVDL","AVDR"]))
actconn_AVDL_AVDR = actconn[AVDL_h,AVDR_h]

#############################################
# COUNT ESYN PAIRS BETWEEN PHARYNX AND OTHERS
#############################################
where_esyn = np.where(esyn)
in_out_pharynx = 0.0
tot_esyn = 0.0
for k in np.arange(len(where_esyn[0])):
    i = where_esyn[0][k]
    j = where_esyn[1][k]
    if i == j: continue
    tot_esyn += 1.0
    if funa.head_ids[j] in funa.pharynx_ids or funa.head_ids[i] in funa.pharynx_ids:
        in_out_pharynx += 1.0
        
in_out_pharynx /= tot_esyn
in_out_pharynx_all = np.sum(~np.isnan(q_wt[funa.pharynx_ai]))/np.sum(~np.isnan(q_wt[funa.head_ai]))
in_out_pharynx_all = in_out_pharynx_all**2

#############################################
# PRINT STUFF
#############################################

print("\n\n")
print("Chem_th=0,gap_th=0")
print("")
print("Fraction of estimated extrasynaptic connections that are in the 1-hop connectome", np.sum(esyn_but_in_aconn),np.sum(esyn),np.sum(esyn_but_in_aconn)/np.sum(esyn))
print("Fraction of fconn that are in the 1-hop connectome, for comparison", np.sum(fconn_in_aconn), np.sum(in_wt), np.sum(fconn_in_aconn)/np.sum(in_wt))
print("Fraction of new in unc31 in the 1-hop connectome",np.sum(new_in_unc31_in_aconn), np.sum(new_in_unc31), np.sum(new_in_unc31_in_aconn)/np.sum(new_in_unc31))

print("List of high-confidence candidate extrasynaptic pairs (only showing excitatory)")
f = open("/projects/LEIFER/francesco/funatlas/figures/paper/figS_esyn/esyn_heatraster.sh","w")
heatraster = "python /home/frandi/dev/pump-probe-analysis/scripts/Andy/heat_raster_plot.py --inclall-occ --sort-avg --relative --vmax:1.2 --headless --nomerge --paired --req_auto_response --matchless-nan-th:0.5 --matchless-nan-th-added-only --dst:/projects/LEIFER/francesco/funatlas/figures/paper/figS_esyn/"
where_esyn = np.where(esyn)
occ3_esyn = []
dff_esyn = []
ids_esyn = []
for k in np.arange(len(where_esyn[0])):
    i = where_esyn[0][k]
    j = where_esyn[1][k]
    occ3_esyn.append(occ3_wt[i,j])
    dff_esyn.append(intensity_map_wt[i,j])
    ids_esyn.append(funa.head_ids[j]+">"+funa.head_ids[i])
    if i == j: continue
    print(funa.head_ids[j]+"->"+funa.head_ids[i])#+"\timap: "+str(np.around(intensity_map_wt[i,j],1)))
    f.write(heatraster+" -j:"+funa.head_ids[j]+" -i:"+funa.head_ids[i]+"\n")
    f.write(heatraster+" -j:"+funa.head_ids[j]+" -i:"+funa.head_ids[i]+" --unc31\n")
f.close()
    
print("List of STRICT high-confidence candidate extrasynaptic pairs (only showing excitatory)")
where_esyn_strict = np.where(esyn_strict)
for k in np.arange(len(where_esyn_strict[0])):
    i = where_esyn_strict[0][k]
    j = where_esyn_strict[1][k]
    if i == j: continue
    print(funa.head_ids[j]+"->"+funa.head_ids[i])
    
print("List of high-confidence candidate autocrine neurons (only showing excitatory)")
where_esyn = np.where(esyn)
autocrine = []
autocrine_ids = []
for k in np.arange(len(where_esyn[0])):
    i = where_esyn[0][k]
    j = where_esyn[1][k]
    iid_app, = funa.approximate_ids([funa.head_ids[i]],merge_bilateral=True)
    jid_app, = funa.approximate_ids([funa.head_ids[j]],merge_bilateral=True)
    if iid_app != jid_app: continue
    autocrine.append([i,j])
    autocrine_ids.append([funa.head_ids[j],funa.head_ids[i]])
    print(funa.head_ids[j]+"->"+funa.head_ids[i])

autocrine = np.array(autocrine)

print("Fraction of the esyn pairs that go into or out from the pharynx.")
print(in_out_pharynx)
print("Compared to the fraction of into or out from the pharynx for all available data")
print(in_out_pharynx_all)
    
######
# MAPS
######
######
# ESYN
######

fig = plt.figure(1)
gs = fig.add_gridspec(1,10)
ax = fig.add_subplot(gs[0,:9])
cax = fig.add_subplot(gs[0,9:])
vmin, vmax = np.min(intensity_map_wt[esyn]), np.max(intensity_map_wt[esyn])
im = ax.imshow(intensity_map_wt,alpha=esyn.astype(float),cmap="jet",vmin=vmin,vmax=vmax,interpolation="nearest")
pp.make_colorbar(cax,vmin,vmax,0.1,cmap="jet",label="dF/F")
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(funa.head_ids)))
ax.set_yticks(np.arange(len(funa.head_ids)))
ax.set_xticklabels(funa.head_ids,fontsize=5,rotation=90)
ax.set_yticklabels(funa.head_ids,fontsize=5)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)

###################
# ESYN BUT IN ACONN
###################

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.imshow(esyn_but_in_aconn*0.5,cmap="Reds")
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(funa.head_ids)))
ax.set_yticks(np.arange(len(funa.head_ids)))
ax.set_xticklabels(funa.head_ids,fontsize=5,rotation=90)
ax.set_yticklabels(funa.head_ids,fontsize=5)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)

###############
# NEW IN UNC-31
###############

fig = plt.figure(3)
gs = fig.add_gridspec(1,10)
ax = fig.add_subplot(gs[0,:9])
cax = fig.add_subplot(gs[0,9:])
vmin, vmax = np.min(intensity_map_unc31[new_in_unc31]), np.max(intensity_map_unc31[new_in_unc31])
im = ax.imshow(intensity_map_unc31,alpha=new_in_unc31.astype(float),vmin=vmin,vmax=vmax,cmap="jet",interpolation="nearest")
pp.make_colorbar(cax,vmin,vmax,0.1,cmap="jet",label="dF/F")
ax.set_xlabel("stimulated",fontsize=30)
ax.set_ylabel("responding",fontsize=30)
ax.set_xticks(np.arange(len(funa.head_ids)))
ax.set_yticks(np.arange(len(funa.head_ids)))
ax.set_xticklabels(funa.head_ids,fontsize=5,rotation=90)
ax.set_yticklabels(funa.head_ids,fontsize=5)
ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)

fig = plt.figure(11)
ax = fig.add_subplot(111)
ax.hist(intensity_map_wt[esyn],bins=30,alpha=0.5,label="estimated esyn")
ax.hist(intensity_map_wt[esyn_but_in_aconn],bins=30,alpha=0.5,label="esyn but in aconn")
ax.hist(intensity_map_unc31[new_in_unc31],bins=30,alpha=0.5,label="new in unc31")
ax.set_xlabel("dF/F")
ax.set_ylabel("counts")
ax.legend()
fig.tight_layout()

############
# HISTOGRAMS
############
#######
# aconn
#######

density = False
nbins = 30
dist1 = aconn[np.logical_and(funa.reduce_to_head(in_wt),~ondiag)]
dist2 = aconn[np.logical_and(esyn,~ondiag)]
_, p_aconn_aconn_esyn = kstest(dist1,dist2,alternative="less")
print("KS p of CDF[in wt aconn] being less than CDF[aconn[esyn]]",p_aconn_aconn_esyn)
fig = plt.figure(12)
ax = fig.add_subplot(111)
_,bins,_=ax.hist(dist1,density=density,bins=nbins,alpha=0.5,label="all")
ax.hist(dist2,density=density,bins=bins,alpha=0.5,label="est. extrasynaptic")
ax.set_yscale("log")
ax.set_xlabel("synaptic count")
ax.set_ylabel("number")
ax.legend()
fig.tight_layout()

#########
# actconn
#########

dist1 = actconn[np.logical_and(funa.reduce_to_head(in_wt),~ondiag)]
dist2 = actconn[np.logical_and(esyn,~ondiag)]
_, p_actconn_actconn_esyn_less = kstest(dist1,dist2,alternative="less")
print("KS p of CDF[in wt actconn] being less than CDF[actconn[esyn]]",p_actconn_actconn_esyn_less)
_, p_actconn_actconn_esyn = kstest(dist1,dist2,alternative="greater")
print("KS p of CDF[in wt actconn] being greater than CDF[actconn[esyn]]",p_actconn_actconn_esyn)
_, p_actconn_actconn_esyn = kstest(dist1,dist2,alternative="two-sided")
print("KS p of CDF[in wt actconn] being two-sided than CDF[actconn[esyn]]",p_actconn_actconn_esyn)
esyn_frac_below_th = np.sum(dist2<split_distr_th)/dist2.shape[0]
all_frac_below_th = np.sum(dist1<split_distr_th)/dist1.shape[0]
print("Fraction of actconn[in wt] below the split_distr_th", all_frac_below_th)
print("Fraction of actconn[esyn] below the split_distr_th", esyn_frac_below_th)

stars = pp.p_to_stars(p_actconn_actconn_esyn_less)

fig = plt.figure(13,figsize=(4,3))
ax = fig.add_subplot(111)
axb = ax.twinx()
nbins = np.logspace(-5,np.log10(np.max(dist1)),30)
n1,bins,_=ax.hist(dist1,density=density,bins=nbins,alpha=0.5,label="all WT",color=c_in_wt)
n2,_,_ = ax.hist(dist2,density=density,bins=bins,alpha=0.5,color=c_esyn,label="candidate extrasynaptic")
axb.step(bins[1:],np.cumsum(n1)/dist1.shape[0],ls="-",color=c_in_wt)
axb.step(bins[1:],np.cumsum(n2)/dist2.shape[0],ls="-",color=c_esyn)
ax.axvline(split_distr_th,c="k",label="Th. from Fig. 3b")
#ax.set_xticks([1e-4,1e-2,1e0])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Anatomy-derived response (V)\nbiophysical model")
ax.set_ylabel("number")
axb.set_ylabel("CDF")
ax.legend(bbox_to_anchor=(1.2,1), loc="upper left").get_frame().set_alpha(0.3)
#ax.text(200,9,"* Candidate extrasynaptic\npairs have anatomy-derived\nresponses smaller than\nall connected pairs\n(p<0.05 one-sided KS test)",fontsize=10,verticalalignment="top")
#fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS_esyn/estimated_extrasynaptic_hist.pdf",bbox_inches="tight")

##################
# actconn in unc31
##################

dist1 = actconn[~ondiag]
dist2 = actconn[np.logical_and(funa.reduce_to_head(in_unc31),~ondiag)]
_, p_actconn_actconn_inunc31 = kstest(dist1,dist2,alternative="less")
print("KS p of CDF[actconn] being less than CDF[actconn[in_unc31]]",p_actconn_actconn_inunc31)
fig = plt.figure(14)
ax = fig.add_subplot(111)
_,bins,_=ax.hist(dist1,density=density,bins=nbins,alpha=0.5,label="all")
ax.hist(dist2,density=density,bins=bins,alpha=0.5,label="est. anatomical")
ax.axvline(split_distr_th,c="k",label="threshold from Fig. 3b")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("anatomy-derived responses")
ax.set_ylabel("number")
ax.legend()
fig.tight_layout()

#########################
# bilateral and autocrine
#########################

fig = plt.figure(15)
ax = fig.add_subplot(111)
for ia in np.arange(autocrine.shape[0]):
    i,j=autocrine[ia]
    aca = actconn[i,j]
    albl = autocrine_ids[ia][0]+"->"+autocrine_ids[ia][1]
    ax.axvline(aca, c="C"+str(ia+1), label=albl, ls="--")
nbins = np.logspace(-5,np.log10(np.max(dist1)),30)
ax.hist(actconn_bilateral[~np.isnan(actconn_bilateral)],bins=nbins,label="all bilateral")
#ax.axvline(np.median(actconn_bilateral[~np.isnan(actconn_bilateral)]),c="C0",label="median bilateral")
ax.axvline(split_distr_th,c="k",label="threshold from Fig. 3b")
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("anatomy-derived responses")
ax.set_ylabel("bilateral pairs")
ax.legend()
fig.tight_layout()

###########################################################################
# NUMBER OF PAIRS IN ESY AS A FUNCTION OF not_in_unc31 FALSE-DISCOVERY RATE
###########################################################################
fig = plt.figure(16)
ax = fig.add_subplot(111)
axb = ax.twinx()
ax.plot(tost_q_th,n_esyn,c="C0",marker="o")
axb.plot(tost_q_th,p_actconn_actconn_esyn_,c="C1",marker="o")
axb.axhline(0.05,color="k")
n_opt = n_esyn[np.argmin(p_actconn_actconn_esyn_)]
ax.axhline(n_opt,color="k")
ax.axvline(tost_q_th[np.argmin(p_actconn_actconn_esyn_)],color="k")
ax.set_yticks(list(ax.get_yticks()) + [n_opt])
axb.set_yticks([0,0.05,0.2,0.4,0.6,0.8,1])
ax.set_ylim(0,None)
ax.set_xlabel("unc-31 TOST q threshold")
ax.set_ylabel("number of extrasyn. pairs")
axb.set_ylabel("p of actonn[esyn]\nsmaller than actconn[in_wt]")
fig.tight_layout()

############
# TIMESCALES
############
density = False
nbins = 30
dist1 = avg_rise_times[np.logical_and(funa.reduce_to_head(in_wt),~ondiag)]
dist2 = avg_rise_times[np.logical_and(esyn,~ondiag)]
_, p_avg_rise_times_esyn = kstest(dist1,dist2,alternative="less")
print("KS p of CDF[avg_rise_times[in_wt]] being less than CDF[avg_rise_times[esyn]]",p_avg_rise_times_esyn)
fig = plt.figure(17)
ax = fig.add_subplot(111)
_,bins,_=ax.hist(dist1,density=density,bins=nbins,alpha=0.5,label="all")
ax.hist(dist2,density=density,bins=bins,alpha=0.5,label="est. extrasynaptic")
ax.set_yscale("log")
ax.set_xlabel("avg_rise_times")
ax.set_ylabel("number")
ax.legend()
fig.tight_layout()

##########################
# SCREENING ILLUSTRATION 1
##########################
fig = plt.figure(18)
ax = fig.add_subplot(111)
ax.plot(np.ravel(q_wt),np.ravel(tost_q_unc31),'o',c="C0",alpha=0.2)
ax.plot((0,0.05,0.05),(tost_q_unc31_th,tost_q_unc31_th,0.0),c="k",alpha=0.5)
ax.set_xlabel("q WT")
ax.set_ylabel(r"q$_{eq}$ $unc-31$")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS_esyn/estimated_extrasynaptic_screen1.pdf",bbox_inches="tight")

fig = plt.figure(19)
ax = fig.add_subplot(111)
ax.plot(occ3_esyn,dff_esyn,'o',c="C0",alpha=0.2)
for i in np.arange(len(ids_esyn)):
    pair = ids_esyn[i]
    x = occ3_esyn[i]
    y = dff_esyn[i]
    ax.text(x,y,pair,fontsize=3)
ax.set_xlabel("Number of observations")
ax.set_ylabel(r"$\langle\Delta F/F\rangle$")
fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS_esyn/estimated_extrasynaptic_screen2.pdf",bbox_inches="tight")


plt.show()
