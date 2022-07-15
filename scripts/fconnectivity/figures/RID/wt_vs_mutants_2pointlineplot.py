import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 39, "smooth_poly": 1,
                 "matchless_nan_th": 0.3}
                 
funa_wt = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=True,merge_AWC=True,
                ds_exclude_tags="mutant", #ds_tags="D20",
                signal_kwargs=signal_kwargs)
                
funa_unc31 = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=True,merge_AWC=True,
                ds_tags="unc31",
                signal_kwargs=signal_kwargs)
                
_, occ2_wt = funa_wt.get_occurrence_matrix(inclall=True)
_, occ2_unc31 = funa_unc31.get_occurrence_matrix(inclall=True)

time = np.linspace(0,60,120)

dFF_wt = funa_wt.get_max_deltaFoverF(occ2_wt,time)
dFF_unc31 = funa_unc31.get_max_deltaFoverF(occ2_unc31,time)

ai_RID = funa_wt.ids_to_i("RID")
ids_i = ["AVE","AVD","AVJ","AWB","RMDV","RIV"]
ai_is = funa_wt.ids_to_i(ids_i)

wt = []
wt_std = []
unc31 = []
unc31_std = []
for ai_i in ai_is:
    wt.append(np.average(np.abs(dFF_wt[ai_i,ai_RID])))
    wt_std.append(np.std(np.abs(dFF_wt[ai_i,ai_RID])))
    unc31.append(np.average(np.abs(dFF_unc31[ai_i,ai_RID])))
    unc31_std.append(np.std(np.abs(dFF_unc31[ai_i,ai_RID])))
    
y = np.array([wt,unc31])
err = np.array([wt_std,unc31_std])

fig = plt.figure(1)
ax = fig.add_subplot(111)
for i in np.arange(len(ai_is)):
    ax.errorbar([0,1],y[:,i],yerr=err[:,i],label=ids_i[i],capsize=5)

ax.set_xticks([0,1])
ax.set_xticklabels(["WT","unc-31"])
ax.set_ylabel("$<max(\Delta F/F)>$")
ax.set_title("Average peak $\Delta$F/F in response to RID stimulation")
ax.legend()
plt.show()
