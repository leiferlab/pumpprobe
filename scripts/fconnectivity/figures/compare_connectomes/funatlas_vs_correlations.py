import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

'''
Things that people do and are wrong, or that don't work in our data:
- q vs connectome (*by q here I mean q2=1-q)
- spontaneous correlation vs connectome (Yemini et al.)

It starts becoming better if you do things properly and simulate the activity:
- q vs simulated-activty connectome
- pharynx: q vs simulated-activity connectome

It becomes much better if you do things properly and look at real activity, and
not simulated activity:
- qsym vs spontaneous correlation
- qsym vs simulated-activity correlation as a comparison that performs worse 
  than with the real activity
'''

to_paper = "--to-paper" in sys.argv
unc31 = "--unc31" in sys.argv
strain = "" if not unc31 else "unc31"
ds_tags = None if not unc31 else "unc31"
ds_exclude_tags = "mutant" if not unc31 else None

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_list_spont = "/projects/LEIFER/francesco/funatlas_list_spont.txt"

                  
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": True}
                 

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=False,
                                 ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                                 verbose=False)
                                 
escon = funa.get_esconn()
#escon = funa.get_effective_esconn()
                                 
# Get the qvalues
occ1,occ2 = funa.get_occurrence_matrix(req_auto_response=True)
occ3 = funa.get_observation_matrix(req_auto_response=True)
_,inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
q,p = funa.get_kolmogorov_smirnov_q(inclall_occ2,return_p=True,strain=strain)
qmax = np.nanmax(q)
q2 = 1.-q
q3 = 1.-q/qmax
qsym = funa.corr_from_eff_causal_conn(q2)
q_ph = funa.reduce_to_pharynx(q)
q2_ph = funa.reduce_to_pharynx(q2)
q3_ph = funa.reduce_to_pharynx(q3)
p_ph = funa.reduce_to_pharynx(p)

# Get the labels confidences
conf, _ = funa.get_labels_confidences(inclall_occ2)

#I2 = funa.ids_to_i("I2")
#AWB = funa.ids_to_i("AWB")
#AVG = funa.ids_to_i("AVG")
#ASK = funa.ids_to_i("ASK")
# Get the kernel-derived correlations
#ck = funa.get_correlation_from_kernels(occ2,occ3,q)#,js=[I2,AWB,AVG,ASK])
km = funa.get_kernels_map(occ2,occ3,filtered=True,include_flat_kernels=True)
#km[q>0.1] = np.nan
#km[occ3<4]=np.nan
print("q>0.05")
km[q>0.05]=np.nan
print("GETTING KERNELS WITH OCC2 INSTEAD OF inclall_OCC2")
ck = funa.get_correlation_from_kernels_map(km,occ3,set_unknown_to_zero=True)
#ck[occ3<4]=np.nan # 4 FIXME FIXME
ck[q>0.05]=np.nan
# RESIMMETRIZE
print("\n\nRESIMMETRIZING CK\n\n")
nanmask = np.isnan(ck)*np.isnan(ck.T)
ck = 0.5*(np.nansum([ck,ck.T],axis=0))
ck[nanmask] = np.nan
ck_orig = np.copy(funa.reduce_to_head(ck))
#ck = funa.get_correlation_from_kernels_full_trace(inclall_occ2,occ3,q)

# Get the signal correlation
stimcorr = funa.get_signal_correlations()
#stimcorr = funa.get_responses_correlations(inclall_occ2,(-10.,30.))

# Load Funatlas for spontaneous activity                                 
funa_spont = pp.Funatlas.from_datasets(ds_list_spont, merge_bilateral=True,
                                         signal="green",
                                         signal_kwargs = signal_kwargs)
# Get the spontaneous activity correlation
spontcorr = funa_spont.get_signal_correlations()
spontcorr_orig = np.copy(funa.reduce_to_head(spontcorr))

# Make anatomical connectome from Funatlas
conn = funa.aconn_chem + funa.aconn_gap
connsym = funa.corr_from_eff_causal_conn(conn)

# Load simulated-activity connectome and corresponding correlations
acfolder = "/projects/LEIFER/francesco/simulations/activity_connectome_sign2/"
actconn = np.loadtxt(acfolder+"activity_connectome_bilateral_merged.txt")
actconncorr = np.loadtxt(acfolder+"activity_connectome_bilateral_merged_correlation.txt")
actconncorr_orig = np.copy(funa.reduce_to_head(actconncorr))
actconn_ph = funa.reduce_to_pharynx(actconn)

print("\n\nRESIMMETRIZING actconncorr\n\n")
nanmask = np.isnan(actconncorr)*np.isnan(actconncorr.T)
actconncorr = 0.5*(np.nansum([actconncorr,actconncorr.T],axis=0))
actconncorr[nanmask] = np.nan

# Prepare array to exclude elements on the diagonal
ondiag = np.zeros_like(q2,dtype=bool)
np.fill_diagonal(ondiag,True)

#fig = plt.figure(1)
#ax = fig.add_subplot(111)
#ax.hist(spontcorr[(~np.isnan(spontcorr))+ondiag],bins=50)
#plt.show()

excl = np.isnan(connsym)+np.isnan(actconncorr)+ondiag
r_connsym_actconncorr = np.corrcoef([connsym[~excl],actconncorr[~excl]])[0,1]

'''spontaneous correlation vs connectome (Yemini et al.)'''
excl = np.isnan(spontcorr)+ondiag
r_spontcorr_conn = np.corrcoef([spontcorr[~excl],conn[~excl]])[0,1]

'''q vs connectome'''
excl = np.isnan(q2)+ondiag
r_q2_conn = np.corrcoef([q2[~excl],conn[~excl]])[0,1]
r_q3_conn = np.corrcoef([q3[~excl],conn[~excl]])[0,1]

'''q vs simulated-activty connectome'''
excl = np.isnan(q2)+np.isnan(actconn)+ondiag
r_q2_actconn = np.corrcoef([q2[~excl],actconn[~excl]])[0,1]
r_q3_actconn = np.corrcoef([q3[~excl],actconn[~excl]])[0,1]

'''pharynx: q vs simulated-activty connectome'''
excl = np.isnan(q2_ph)+np.isnan(actconn_ph)+funa.reduce_to_pharynx(ondiag)
r_q2_actconn_ph = np.corrcoef([q2_ph[~excl],actconn_ph[~excl]])[0,1]
r_q3_actconn_ph = np.corrcoef([q3_ph[~excl],actconn_ph[~excl]])[0,1]

'''qsym vs spontaneous correlation'''
excl = np.isnan(qsym)+np.isnan(spontcorr)+ondiag#+(spontcorr<0)
r_qsym_spontcorr = np.corrcoef([qsym[~excl],spontcorr[~excl]])[0,1]

'''qsym vs simulated-activity correlation'''
excl = np.isnan(qsym)+np.isnan(actconncorr)+ondiag
r_qsym_actconncorr = np.corrcoef([qsym[~excl],actconncorr[~excl]])[0,1]

# HAVE ONE excl FOR ALL THE FOLLOWING COMPARISONS
excl = np.isnan(actconncorr)+np.isnan(spontcorr)+np.isnan(ck)+np.isnan(stimcorr)+np.isnan(q2)+ondiag
excl_ = np.copy(excl)

'''ck vs spontaneous correlation'''
spontcorr_ = spontcorr[~excl]
ck_ = ck[~excl]
weights_ = (np.nanmax(q)-q)[~excl]#q2[~excl]#np.ones_like(ck_)
r_spontcorr_ck = np.corrcoef([spontcorr_,ck_])[0,1]
rs_spontcorr_ck = pp.pearsonr_sample(spontcorr_,ck_)
rw_spontcorr_ck = pp.weighted_corr(spontcorr_,ck_,weights_)
A = np.array([spontcorr_,]).T
par, res, _, _ = np.linalg.lstsq(A,ck_,rcond=None)
m, = par
c = 0
#par,res,_,_,_ = np.polyfit(spontcorr_,ck_,1,full=True)
#m,c=par
avg_ck_ = np.average(ck_)
SSres = np.sum( np.power(ck_ - (m*spontcorr_+c),2) )
SStot = np.sum( np.power(ck_ - avg_ck_,2) )
R2_spontcorr_ck = 1.0 - SSres/SStot


x_spontcorr_ck = np.array([np.min(spontcorr_),np.max(spontcorr_)])
l_spontcorr_ck = m*x_spontcorr_ck+c

'''simulated-activity correlation vs spontaneous correlation'''
actconncorr_ = actconncorr[~excl]
r_spontcorr_actconncorr = np.corrcoef([spontcorr_,actconncorr_])[0,1]
rs_spontcorr_actconncorr = pp.pearsonr_sample(spontcorr_,actconncorr_)
rw_spontcorr_actconncorr = pp.weighted_corr(spontcorr_,actconncorr_,weights_)
A = np.array([spontcorr_]).T
par, res, _, _ = np.linalg.lstsq(A,actconncorr_,rcond=None)
m, = par
c = 0
#par,res,_,_,_ = np.polyfit(spontcorr_,actconncorr_,1,full=True)
#m,c=par

avg_actconncorr_ = np.average(actconncorr_)
SSres = np.sum( np.power(actconncorr_ - (m*spontcorr_+c),2) )
SStot = np.sum( np.power(actconncorr_ - avg_actconncorr_,2) )
R2_spontcorr_actconncorr = 1.0 - SSres/SStot

x_spontcorr_actconncorr = np.array([np.min(spontcorr_),np.max(spontcorr_)])
l_spontcorr_actconncorr = m*x_spontcorr_actconncorr+c

'''stimcorr vs spontaneous correlation'''
#excl = np.isnan(spontcorr)+np.isnan(stimcorr)+ondiag
r_spontcorr_stimcorr = np.corrcoef([spontcorr[~excl],stimcorr[~excl]])[0,1]

print('''spontaneous correlation vs connectome (Yemini et al.)''',r_spontcorr_conn)
print('''q vs connectome''',r_q2_conn,r_q3_conn)
print('''q vs simulated-activty connectome''',r_q2_actconn,r_q3_actconn)
print('''q vs simulated-activty connectome (pharynx)''',r_q2_actconn_ph,r_q3_actconn_ph)
print('''qsym vs simulated-activity correlation''',r_qsym_actconncorr)
print('''qsym vs spontaneous correlation''',r_qsym_spontcorr)
print('''connectome-derived correlation vs spontaneous correlation''',r_spontcorr_actconncorr,rw_spontcorr_actconncorr,R2_spontcorr_actconncorr,rs_spontcorr_actconncorr)
#print(r_connsym_actconncorr)
print('''kernel-derived correlation vs spontaneous correlation''',r_spontcorr_ck,rw_spontcorr_ck,R2_spontcorr_ck,rs_spontcorr_ck)
print('''stimcorr vs spontaneous correlation''',r_spontcorr_stimcorr)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
bars = [r_spontcorr_ck,
        r_spontcorr_actconncorr,
        r_spontcorr_conn,
        ]
x = np.arange(len(bars))[::-1]/2
ax.bar(x,bars,width=0.4,align="center")
ax.set_xlim(0,0.5)
'''ax.set_ylabel("Correlation coefficient")
ax.set_yticks(y)
ax.set_yticklabels(["FunConn-derived\nactivity correlations",
                    "Anatomy-derived\nactivity correlations",
                    "Anatomical weight\n(synaptic count)",
                    ],
                    rotation=0,va="center")'''
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig1.tight_layout()
if not unc31:
    fig1.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations3_nth.png",dpi=300,bbox_inches="tight")
    if to_paper:
        fig1.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_correlations_nth.pdf",bbox_inches="tight")
    
fig1b = plt.figure(11)
ax = fig1b.add_subplot(111)
bars = [r_q2_actconn,
        r_q2_actconn_ph,
        ]
y = np.arange(len(bars))[::-1]/2
ax.barh(y,bars,height=0.4,align="center")
ax.set_xlim(0,0.5)
ax.set_xlabel("Correlation coefficient")
ax.set_yticks(y)
ax.set_yticklabels(["(1-q)\nanatomy-derived effective weight",
                    "pharynx (1-q)\nanatomy-derived effective weight",
                    ],
                    rotation=0,va="center")
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
fig1b.tight_layout()
if not unc31:
    fig1b.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations3b_nth.png",dpi=300,bbox_inches="tight")
    if to_paper:
        fig1b.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_correlationsb_nth.pdf",bbox_inches="tight")
    
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
excl = np.isnan(qsym)+np.isnan(spontcorr)+ondiag
ax2.plot(spontcorr[~excl],qsym[~excl]-np.min(qsym[~excl]),'o',alpha=0.3)
ax2.axvline(0,c="k",alpha=0.5)
ax2.axhline(0,c="k",alpha=0.5)
ax2.set_xlabel(r"$\langle$spontaneous correlation$\rangle_{ds}$")
ax2.set_ylabel("\"symmetrized\" (1-q)")
fig2.tight_layout()
fig2.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations3_r_qsym_spontcorr.png",dpi=300,bbox_inches="tight")
if to_paper:
    #fig2.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_correlations_r_qsym_spontcorr.pdf",bbox_inches="tight")
    pass
    
fig2b = plt.figure(12)
ax2b = fig2b.add_subplot(111,projection='3d')
excl = np.isnan(ck)+np.isnan(spontcorr)+np.isnan(connsym)+ondiag
ax2b.scatter(ck[~excl],connsym[~excl],spontcorr[~excl],alpha=0.3)
#ax2b.scatter(qsym[~excl]-np.min(qsym[~excl]),connsym[~excl],spontcorr[~excl])
ax2b.set_xlabel("ck")
ax2b.set_ylabel("connsym")
ax2b.set_zlabel("spontcorr")

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
excl = np.isnan(q)+np.isnan(actconn)#+ondiag
cm = np.array(["C0","C0"])#np.array(["y","C0"])
escon_ = 1*escon
ax3.scatter(np.abs(actconn[~excl]),q[~excl],c=cm[escon_[~excl]],alpha=0.3)
ax3.set_xlabel("anatomy-derived effective weight (abs.value) $|\Delta V_{i,j}|$  (V)")
ax3.set_ylabel("q")
ax3.invert_yaxis()
fig3.tight_layout()
if not unc31:
    fig3.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations3_r_q_actconn.png",dpi=300,bbox_inches="tight")
    if to_paper:
        fig3.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_correlations_r_q_actconn.pdf",bbox_inches="tight")
        pass
    
fig3b = plt.figure(13)
ax3b = fig3b.add_subplot(111)
excl = np.isnan(q)+np.isnan(actconn)+ondiag
cm = np.array(["C0","C0"])#np.array(["y","C0"])
escon_ = 1*escon
ax3b.scatter(np.abs(actconn[~excl]),q[~excl],c=cm[escon_[~excl]],alpha=0.3)
ax3b.set_xlabel("anatomy-derived effective weight (abs.value) $|\Delta V_{i,j}|$ (V)")
ax3b.set_ylabel("q")
ax3b.invert_yaxis()
fig3b.tight_layout()
if not unc31:
    fig3b.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations3_r_q_actconn_no_diag.png",dpi=300,bbox_inches="tight")
    if to_paper:
        fig3b.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_correlations_r_q_actconn_no_diag.pdf",bbox_inches="tight")
        
fig3c = plt.figure(23)
ax3c = fig3c.add_subplot(111)
excl = np.isnan(q)+np.isnan(actconn)+ondiag
ax3c.hist(actconn[~escon],alpha=0.3,color=cm[0],density=True,range=(0,0.007),bins=30)
ax3c.hist(actconn[escon],alpha=0.3,color=cm[1],density=True,range=(0,0.007),bins=30)
ax3c.set_xlabel("anatomy-derived effective weight")
ax3c.set_ylabel("density")
fig3c.tight_layout()

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
excl = np.isnan(q_ph)+np.isnan(actconn_ph)+funa.reduce_to_pharynx(ondiag)
cm = np.array(["C0","C0"])#np.array(["y","C0"])
escon_ = 1*escon
escon_ph_ = funa.reduce_to_pharynx(escon_)
ax4.scatter(actconn_ph[~excl],q_ph[~excl],c=cm[escon_ph_[~excl]],alpha=0.3)
ax4.set_xlabel("anatomy-derived effective weight (pharynx only)")
ax4.set_ylabel("q")
ax4.invert_yaxis()
fig4.tight_layout()
if not unc31:
    fig4.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/funatlas_vs_correlations3_r_q_actconn_ph.png",dpi=300,bbox_inches="tight")
    if to_paper:
        fig4.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/funatlas_vs_correlations_r_q_actconn_ph.pdf",bbox_inches="tight")

# HAVE A SINGLE excl
excl = np.isnan(actconncorr)+np.isnan(spontcorr)+np.isnan(ck)+np.isnan(stimcorr)+np.isnan(q2)+ondiag

fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)
#excl = np.isnan(ck)+np.isnan(spontcorr)+ondiag
cm = np.array(["C0","C0"])#np.array(["y","C0"])
escon_ = 1*escon
ax5.scatter(spontcorr_,ck_,c=cm[escon_[~excl_]],alpha=0.3)
ax5.plot(x_spontcorr_ck,l_spontcorr_ck)
ax5.axvline(0,c="k",alpha=0.5)
ax5.axhline(0,c="k",alpha=0.5)
ax5.set_xlabel(r"$\langle$spontaneous correlation$\rangle_{ds}$")
ax5.set_ylabel("kernel-derived correlation")
if not unc31:
    if to_paper:
        fig5.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kernel_vs_spont.pdf",dpi=300,bbox_inches="tight")
fig5.tight_layout()

fig6= plt.figure(6)
ax6 = fig6.add_subplot(111)
#excl = np.isnan(actconncorr)+np.isnan(spontcorr)+ondiag
cm = np.array(["C0","C0"])#np.array(["y","C0"])
escon_ = 1*escon
ax6.scatter(spontcorr_,actconncorr_,c=cm[escon_[~excl_]],alpha=0.3)
ax6.plot(x_spontcorr_actconncorr,l_spontcorr_actconncorr)
ax6.axvline(0,c="k",alpha=0.5)
ax6.axhline(0,c="k",alpha=0.5)
ax6.set_xlabel(r"$\langle$spontaneous correlation$\rangle_{ds}$")
ax6.set_ylabel("anatomy-derived correlation")
if not unc31:
    if to_paper:
        fig6.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kunert_vs_spont.pdf",dpi=300,bbox_inches="tight")
fig6.tight_layout()

fig7= plt.figure(7)
ax7 = fig7.add_subplot(111)
#excl = np.isnan(stimcorr)+np.isnan(spontcorr)+ondiag
cm = np.array(["C0","C0"])#np.array(["y","C0"])
escon_ = 1*escon
ax7.scatter(spontcorr[~excl],stimcorr[~excl],c=cm[escon_[~excl_]],alpha=0.3)
ax7.axvline(0,c="k",alpha=0.5)
ax7.axhline(0,c="k",alpha=0.5)
ax7.set_xlabel(r"|$\langle$spontaneous correlation$\rangle_{ds}$|")
ax7.set_ylabel("bare correlation in datasets")
fig7.tight_layout()

###
# Plot the correlation matrices
###

fig8 = plt.figure(8)
ax8 = fig8.add_subplot(111)
ax8.set_facecolor((0.4,0.4,0.4))
_,sorter_i,sorter_j,lim = funa.sort_matrix_pop_nans(spontcorr_orig,return_all=True)
np.fill_diagonal(spontcorr_orig,np.nan)
ax8.imshow(spontcorr_orig[sorter_i][:,sorter_j])
ax8.set_xticks([])
ax8.set_yticks([])
fig8.tight_layout()
if to_paper:
    fig8.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_spont.pdf",dpi=300,bbox_inches="tight")
    fig8.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_spont.png",dpi=300,bbox_inches="tight")
    
fig9 = plt.figure(9)
ax9 = fig9.add_subplot(111)
ax9.set_facecolor((0.4,0.4,0.4))
np.fill_diagonal(ck_orig,np.nan)
ax9.imshow(ck_orig[sorter_i][:,sorter_j],vmax=0.02)
ax9.set_xticks([])
ax9.set_yticks([])
fig9.tight_layout()
if to_paper and not unc31:
    fig9.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kernels.pdf",dpi=300,bbox_inches="tight")
    fig9.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kernels.png",dpi=300,bbox_inches="tight")
    
fig10 = plt.figure(10)
ax10 = fig10.add_subplot(111)
ax10.set_facecolor((0.4,0.4,0.4))
np.fill_diagonal(actconncorr_orig,np.nan)
ax10.imshow(actconncorr_orig[sorter_i][:,sorter_j])
ax10.set_xticks([])
ax10.set_yticks([])
fig10.tight_layout()
if to_paper and not unc31:
    fig10.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kunert.pdf",dpi=300,bbox_inches="tight")
    fig10.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kunert.png",dpi=300,bbox_inches="tight")

##    
#Sort the correlation matrices
##

sorter2 = np.argnansort(np.nansum(np.arange(spontcorr_orig[sorter_i][:,sorter_j].shape[0])[:,None]*spontcorr_orig[sorter_i][:,sorter_j],axis=0))[::-1]
fig40 = plt.figure(40)
ax = fig40.add_subplot(111)
ax.set_facecolor((0.4,0.4,0.4))
ax.imshow(spontcorr_orig[sorter_i][:,sorter_j][sorter2][:,sorter2])
ax.set_xticks([])
ax.set_yticks([])
fig40.tight_layout()
if to_paper and not unc31:
    fig40.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_spont_sorted.pdf",dpi=300,bbox_inches="tight")
    fig40.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_spont_sorted.png",dpi=300,bbox_inches="tight")

fig41 = plt.figure(41)
ax = fig41.add_subplot(111)
ax.set_facecolor((0.4,0.4,0.4))
ax.imshow(ck_orig[sorter_i][:,sorter_j][sorter2][:,sorter2])
ax.set_xticks([])
ax.set_yticks([])
fig41.tight_layout()
if to_paper and not unc31:
    fig41.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kernels_sorted.pdf",dpi=300,bbox_inches="tight")
    fig41.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kernels_sorted.png",dpi=300,bbox_inches="tight")

fig42 = plt.figure(42)
ax = fig42.add_subplot(111)
ax.set_facecolor((0.4,0.4,0.4))
ax.imshow(actconncorr_orig[sorter_i][:,sorter_j][sorter2][:,sorter2])
ax.set_xticks([])
ax.set_yticks([])
fig42.tight_layout()
if to_paper and not unc31:
    fig42.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kunert_sorted.pdf",dpi=300,bbox_inches="tight")
    fig42.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/corr_kunert_sorted.png",dpi=300,bbox_inches="tight")

np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/sorter_i.txt",sorter_i,fmt="%d")
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/sorter_j.txt",sorter_j,fmt="%d")
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/fig3/sorter_2.txt",sorter2,fmt="%d")

plt.show()

