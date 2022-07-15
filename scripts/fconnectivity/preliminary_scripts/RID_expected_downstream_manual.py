import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp


# DATASETS WITH STIMULATION OF RID
ds_list = [
    "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_101248/",
    "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_152524/"
    ]


funa = pp.Funatlas.from_datasets(
                ds_list,                
                merge_bilateral=True,merge_dorsoventral=False,merge_AWC=True)         

occ1, occ2 = funa.get_occurrence_matrix()
occ3 = funa.get_observation_matrix()    
ai_RID = funa.ids_to_i("RID")

fnames = ["external_data/GenesExpressing-npr-4-thrs2.csv",
          "external_data/GenesExpressing-npr-11-thrs2.csv",
          "external_data/GenesExpressing-pdfr-1-thrs2.csv",
          "external_data/GenesExpressing-daf-2-thrs2.csv"]
          
trans_rec_pairs = ["FLP-14,NPR-4",
                   "FLP-14,NPR-11",
                   "PDF-1,PDFR-1",
                   "INS-17,DAF-2"]
                   
trans_exp_level = np.array([110634.0,110634.0,157972.0,1505.0])

'''excl_neu = [
    # Exclude neurons that are never observed
    "AIA_","RIB_","RIR_","RIF_","AUA_","RIR_","RMG_","RMF_","AIY_","ADF_","ALA","AVB_","RIS_","AVF_","RMH_","RIP_"
    # and that have not been observed in the recordings that I am using
    "ASK_","BAG_","RIC_","RIS_","ADE_","URB_","MC_",
    ]'''

                   
x = np.arange(len(funa.head_ai))
exp_levels = np.zeros((len(fnames),len(funa.head_ai)))
for i_f in np.arange(len(fnames)):
    f = open(fnames[i_f],"r")
    lines = f.readlines()
    f.close()
    
    exp_levels_ = np.zeros(funa.n_neurons)
    
    
    for line in lines[1:]:
        s = line.split(",")
        cell_id = s[1][1:-1]
        exp_level = float(s[2])
        
        ai = funa.ids_to_i(cell_id)
        exp_levels_[ai] = exp_level*trans_exp_level[i_f]
    
    
    b = funa.reduce_to_head(exp_levels_)
    exp_levels[i_f] = b
    
# Sort them by the sum of the expression levels
tot_exp_levels = np.sum(exp_levels,axis=0)
bargsort = np.argsort(tot_exp_levels)[::-1]
lim = np.where(tot_exp_levels[bargsort]==0)[0][0]

fig = plt.figure(1,figsize=(16,9))
ax = fig.add_subplot(111)
x = np.arange(len(funa.head_ai))

prev_b = np.zeros(len(funa.head_ai))
for i_f in np.arange(len(fnames)):
    b = exp_levels[i_f][bargsort]
    
    # Set to zero the neurons that are never observed
    for i_hn in np.arange(len(funa.head_ids)):
        id_hn = str(funa.head_ids[bargsort][i_hn])
        ai_hn = funa.ids_to_i(id_hn)
        if occ3[ai_hn,ai_RID]==0:
            b[i_hn] = 0
    
    ax.bar(x,b,label=trans_rec_pairs[i_f],bottom=prev_b)
    prev_b += b

#ax.set_xticks(x[:lim])
ax.set_xticks(x)
#ax.set_xticklabels(funa.head_ids[bargsort][:lim],rotation=90,fontsize=9)
ax.set_xticklabels(funa.head_ids[bargsort],rotation=90,fontsize=9)
#ax.set_xlim(-1,lim)
ax.set_ylabel("CeNGEN transmitter count in RID * receptor counts in downstream neurons")
ax.set_title("\"Expected\" amplitude of reponse based on receptor and transmitter \n"+\
             "expression count from CeNGEN (transmitter,receptor)")
ax.legend()

# Manually compiled downstream neurons
'''downst_in_data = [
    # From the first 2D measurement (look it up, it's in the targeting tests)
    "AVD_",
    # From /projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_101248/
    # stim1
    "AVAL","AVEL","RMDVL","AVDL","AWBL","AVER","OLQD",
    # stim2 
    "AVAL","SMDDL","RMDL","AVEL","RMDVL","SMBDL","AVDL","RMDDL","AWBL","RIG","RMDD_","CEPDL","AIMR","RIMR","OLQD","AVJR",
    # From /projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_152524/
    # stim 1 (note that some are inhibitory here, but were excitatory in the other recording. and some pairs are one exc and one inh, like AVE)
    "RMDVL","AVDL","AVJL","M2L","AVEL","I2L","RIG","RIG","AIBL","AVER","RMDVR","AVDR",
    # stim 2 (here both AVER and AVEL are inh)
    "AVDL","M2L","RMEL","AVEL","RIG","IL2VL","AIBL","RIVL","M1","URYDR","AVER","AIBR","RMDR","RMDVR","AVDR",
    ]

downst_in_data = funa.approximate_ids(downst_in_data,merge_bilateral=True,merge_dorsoventral=False,merge_AWC=True)
downst_in_data, downst_in_data_counts = np.unique(downst_in_data,return_counts=True)'''

downst_in_data_ai = np.where(occ1[:,ai_RID]>0)[0]

downst_in_data_ai_head = funa.ai_to_head(downst_in_data_ai)
exp_bars = np.zeros_like(tot_exp_levels)
exp_bars[downst_in_data_ai_head] = occ1[downst_in_data_ai,ai_RID]/occ3[downst_in_data_ai,ai_RID]
exp_bars = exp_bars[bargsort]
new_lim = np.max(np.where(exp_bars!=0)[0])

axb = ax.twinx()
#axb.bar(x,exp_bars,color="#9467bd",alpha=0.6,label="response count")
h_ais_barg = funa.head_ai[bargsort]
for i in np.arange(exp_bars.shape[0]):
    if occ3[h_ais_barg[i],ai_RID]>0:
        axb.plot((i,i),(0,1),c="purple",marker="x",alpha=0.3)
    if exp_bars[i]>0:
        if i==0:
            axb.plot((i,i),(0,exp_bars[i]),c="k",marker="x",label="fractional response count")
        else:
            axb.plot((i,i),(0,exp_bars[i]),c="k",marker="x")

#if new_lim>lim: 
    #ax.set_xlim(-1,new_lim+1)
    #ax.set_xticks(x[:new_lim+1])
    #new_ticklabels = funa.head_ids[bargsort][:new_lim+1].copy()
    #ax.set_xticklabels(new_ticklabels,rotation=90,fontsize=9)
axb.set_ylim(0,)            
axb.set_ylabel("Fraction of times neurons responded in experiments")
axb.legend(loc=5)

fig.tight_layout()
plt.show()
    
