import numpy as np, matplotlib.pyplot as plt, sys, re, matplotlib.gridspec as gridspec
import pumpprobe as pp, wormdatamodel as wormdm

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

jid = "RID"
vmax = 1
    
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
for s in sys.argv:
    sa = s.split(":")
    if sa[0] in ["-i","--i"] : iid=sa[1]
    if sa[0] in ["-j","--j"] : jid=sa[1]
    if sa[0] == "--vmax": vmax=float(sa[1])
    if sa[0] == "--matchless-nan-th": matchless_nan_th=float(sa[1])

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True, 
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only}

funa_wt = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=False,merge_dorsoventral=False,
                merge_numbered=False,merge_AWC=False,
                ds_exclude_tags="mutant",signal="green",
                signal_kwargs=signal_kwargs)
                
funa_unc31 = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=False,merge_dorsoventral=False,
                merge_numbered=False,merge_AWC=False,
                ds_tags="unc31",signal="green",
                signal_kwargs=signal_kwargs)
                
_, occ2_wt = funa_wt.get_occurrence_matrix(inclall=True)
_, occ2_unc31 = funa_unc31.get_occurrence_matrix(inclall=True)

ai_RID = funa_wt.ids_to_i(jid)
ai_ADLR = funa_wt.ids_to_i("ADLR")

t = np.linspace(-10,30,120)
t0 = np.argmin(np.abs(t-0))

fig = plt.figure(1,figsize=(14,3))
#spec = gridspec.GridSpec(ncols=1, nrows=5, figure=fig)
#ax1 = fig.add_subplot(spec[0,0])
#ax2 = fig.add_subplot(spec[1:4,0])
#ax3 = fig.add_subplot(spec[4,0])
#fig.subplots_adjust(hspace=0.0)
ax2 = fig.add_subplot(111)

xticks = []
xticklabels = []

n = 0
dn = 1.0
d = 0.2
bar_width = 0.2

dff_th = 0.03#0.1

screen_th = 0.15
screened = []

to_be_saved = []
for ai_i in np.arange(len(occ2_wt[:,ai_RID])):
    if ai_i == ai_RID: continue
    if len(occ2_wt[ai_i,ai_RID])==0 or len(occ2_unc31[ai_i,ai_RID])==0:
        continue
        
    avg_wt = []
    avg_unc31 = []

    for io in np.arange(len(occ2_wt[ai_i,ai_RID])):
        o = occ2_wt[ai_i,ai_RID][io]
        ds = o["ds"]
        ie = o["stim"]
        i = o["resp_neu_i"]
        
        time,time2,i0,i1 = funa_wt.fconn[ds].get_time_axis(ie,True)
        shift_vol = funa_wt.fconn[ds].shift_vols[ie]
        
        y = funa_wt.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,i]
        nan_mask = funa_wt.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
        
        pre = np.average(y[:shift_vol])
        y = (y-pre)/pre
               
        if np.max(y)>2:continue
            
        if pp.Fconn.nan_ok(nan_mask,pp.Funatlas.nan_th*len(nan_mask)):
            avg_wt.append(np.nanmean(y[shift_vol:shift_vol+40]))
            
    avg_wt = np.array(avg_wt)
    if funa_wt.neuron_ids[ai_i]=="URXL":
        print(occ2_wt[ai_i,ai_RID])
        print(avg_wt)
            
    for io in np.arange(len(occ2_unc31[ai_i,ai_RID])):
        o = occ2_unc31[ai_i,ai_RID][io]
        ds = o["ds"]
        ie = o["stim"]
        i = o["resp_neu_i"]
        
        time,time2,i0,i1 = funa_unc31.fconn[ds].get_time_axis(ie,True)
        shift_vol = funa_unc31.fconn[ds].shift_vols[ie]
        y = funa_unc31.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,i]
        nan_mask = funa_unc31.sig[ds].get_segment_nan_mask(i0,i1)[:,i]
        
        pre = np.average(y[:shift_vol])
        y = (y-pre)/pre
        
        try:
            if np.max(y)>2:continue
        except:
            pass
        
        if pp.Fconn.nan_ok(nan_mask,pp.Funatlas.nan_th*len(nan_mask)):
            avg_unc31.append(np.nanmean(y[shift_vol:shift_vol+40]))
    
    avg_unc31 = np.array(avg_unc31)      
    
    if np.abs(np.nanmean(avg_wt))<dff_th:# and np.abs(np.average(avg_unc31))<dff_th: #This second condition was here by mistake. We want to be able to see connections that disappear in unc-31.
        continue
    
    if np.isnan(np.nanmean(avg_wt)) or np.isnan(np.nanmean(avg_unc31)):
        continue
        
    if np.absolute(np.nanmean(avg_wt)-np.nanmean(avg_unc31))>screen_th:
        screened.append(ai_i)

    ax2.bar(n*dn,np.nanmean(avg_wt),color="C0",width=bar_width,alpha=0.6)
    ax2.bar(n*dn+d,np.nanmean(avg_unc31),color="C1",width=bar_width,alpha=0.6)
    #ax1.bar(n*dn,np.nanmean(avg_wt),color="C0",width=bar_width,alpha=0.6)
    #ax1.bar(n*dn+d,np.nanmean(avg_unc31),color="C1",width=bar_width,alpha=0.6)
    #ax3.bar(n*dn,np.nanmean(avg_wt),color="C0",width=bar_width,alpha=0.6)
    #ax3.bar(n*dn+d,np.nanmean(avg_unc31),color="C1",width=bar_width,alpha=0.6)
    
    #ax1.scatter(n*dn+np.random.random(len(avg_wt))*bar_width - bar_width/2, avg_wt, color="C0",s=0.5)
    #ax1.scatter(n*dn+d+np.random.random(len(avg_unc31))*bar_width - bar_width/2, avg_unc31, color="C1",s=0.5)
    
    ax2.scatter(n*dn+np.random.random(len(avg_wt))*bar_width - bar_width/2, avg_wt, color="C0",s=0.5)
    ax2.scatter(n*dn+d+np.random.random(len(avg_unc31))*bar_width - bar_width/2, avg_unc31, color="C1",s=0.5)
    
    to_be_saved.append([avg_wt,avg_unc31])
    
    #ax3.scatter(n*dn+np.random.random(len(avg_wt))*bar_width - bar_width/2, -avg_wt, color="C0",s=0.5)
    #ax3.scatter(n*dn+d+np.random.random(len(avg_unc31))*bar_width - bar_width/2, -avg_unc31, color="C1",s=0.5)
    
    
    xticks.append(n*dn+d/2)
    xticklabels.append(funa_wt.neuron_ids[ai_i])
    
    n += 1
    
ax2.axhline(0,color="k",lw=1)
ax2.set_xlim(-1,n)
ax2.set_ylim(-0.8,1.2)
ax2.set_yticks([-0.5,0,0.5,1.0])
ax2.set_ylabel(r"$\Delta F/F$")
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels,fontsize=8,rotation=90)
ax2.spines.top.set_visible(False)
#ax2.spines.bottom.set_visible(False)
ax2.spines.right.set_visible(False)
ax2.xaxis.tick_bottom()

'''
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)
ax3.plot([0], [1], transform=ax3.transAxes, **kwargs)

ax1.set_xlim(-1,n)
ax1.set_ylim(1.22,None)
ax1.set_yscale("log")
ax1.set_yticks([2,3,4,5])
ax1.set_yticklabels(["2","","4",""])
ax1.spines.bottom.set_visible(False)
ax1.spines.top.set_visible(False)
ax1.spines.right.set_visible(False)
ax1.tick_params(labeltop=False,bottom=False,top=False)

#ax3.set_xlim(-1,n)
#ax3.set_xticks(xticks)
#ax3.set_xticklabels(xticklabels,fontsize=8,rotation=90)
#ax3.set_ylim(0.6,None)
#ax3.set_yscale("log")
#ax3.invert_yaxis()
#ax3.set_yticks([0.6,0.7,0.8,0.9,1,2,3,4])
#ax3.set_yticklabels(["","","","","","-2","","-4"])
#ax3.spines.top.set_visible(False)
#ax3.spines.right.set_visible(False)
#ax3.xaxis.tick_bottom()'''


print("\n\nUsing dff_th",dff_th,". It was 0.1 in the paper\n\n")

for ai_i in screened:
    print(funa_wt.neuron_ids[ai_i])

#fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS16/RID_bar_plot.pdf",dpi=300,bbox_inches="tight")
#for i in np.arange(len(to_be_saved)):
#    tbs = to_be_saved[i]
#    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS16/RID_bar_plot_"+str(i).zfill(3)+"_A.txt",tbs[0])
#    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS16/RID_bar_plot_"+str(i).zfill(3)+"_B.txt",tbs[1])

f = open("/projects/LEIFER/francesco/funatlas/figures/paper/figS16/RID_bar_plot.txt","w")
for i in np.arange(len(to_be_saved)):
    tbs = to_be_saved[i]
    f.write(xticklabels[i]+" wt,"+",".join(tbs[0].astype(str))+"\n")
    f.write(xticklabels[i]+" unc31,"+",".join(tbs[1].astype(str))+"\n")
f.close()



plt.show()
