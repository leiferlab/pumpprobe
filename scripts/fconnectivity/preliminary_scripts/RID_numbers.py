import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

funa = pp.Funatlas.from_datasets(
                ds_list,                
                merge_bilateral=True,merge_dorsoventral=True,merge_AWC=True,
                ds_exclude_tags="mutant")
                
occ1, occ2 = funa.get_occurrence_matrix()
occ3 = funa.get_observation_matrix()

print(funa.head_ids)
print("SAB#####################")

occ1 = funa.reduce_to_head(occ1)
occ3 = funa.reduce_to_head(occ3)

ai_RID = funa.ai_to_head([funa.ids_to_i("RID")])
ai_AIY = funa.ai_to_head([funa.ids_to_i("AIY")])
occ1[ai_AIY,ai_RID] = 0 # Don't trust that labeling of mine
occ1[ai_RID,ai_RID] = 0
occ3[ai_AIY,ai_RID] = 0 # Don't trust that labeling of mine
occ3[ai_RID,ai_RID] = 0

fnames = ["external_data/GenesExpressing-npr-4-thrs2.csv",
              "external_data/GenesExpressing-npr-11-thrs2.csv",
              "external_data/GenesExpressing-pdfr-1-thrs2.csv",
              #"external_data/GenesExpressing-daf-2-thrs2.csv"
              ]
              
trans_rec_pairs = ["FLP-14,NPR-4",
                   "FLP-14,NPR-11",
                   "PDF-1,PDFR-1",
                   #"INS-17,DAF-2"
                   ]
                   
trans_exp_level = np.array([110634.0,110634.0,157972.0,])#1505.0])

# Build expression levels                   
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
        
        cell_id = funa.cengen_ids_conversion(cell_id)
        
        for cid in cell_id:
            ai = funa.ids_to_i(cid)
            exp_levels_[ai] = exp_level*trans_exp_level[i_f]
        
    b = funa.reduce_to_head(exp_levels_)
    exp_levels[i_f] = b
    
has_any_receptor = np.logical_or(np.logical_or((exp_levels[0]>0),(exp_levels[1]>0)),(exp_levels[2]>0))
print("Observed neurons:",np.sum(occ3[:,ai_RID]>0))
print("Responding neurons:",np.sum(occ1[:,ai_RID]>0))
print("Non-responding neurons:",np.sum((occ3[:,ai_RID]>0)*(occ1[:,ai_RID]==0)))
print("Has any receptor:",np.sum(has_any_receptor))
print("Has npr-4:",np.sum(exp_levels[0]>0))
print("Has npr-11:",np.sum(exp_levels[1]>0))
print("Has pdfr-1:",np.sum(exp_levels[2]>0))
print("Observed with any receptor:",np.sum((occ3[:,ai_RID]>0)[:,0]*has_any_receptor))
print("Observed with npr-4:",np.sum((occ3[:,ai_RID]>0)[:,0]*(exp_levels[0]>0)))
print("Observed with npr-11:",np.sum((occ3[:,ai_RID]>0)[:,0]*(exp_levels[1]>0)))
print("Observed with pdfr-1:",np.sum((occ3[:,ai_RID]>0)[:,0]*exp_levels[2]>0))
print("Responding with any receptor:",np.sum( (occ1[:,ai_RID]>0)[:,0]*has_any_receptor))
print("Responding with npr-4:",np.sum( (occ1[:,ai_RID]>0)[:,0]*(exp_levels[0]>0) ) )
print("Responding with npr-11:",np.sum( (occ1[:,ai_RID]>0)[:,0]*(exp_levels[1]>0)))
print("Responding with pdfr-1:",np.sum( (occ1[:,ai_RID]>0)[:,0]*(exp_levels[2]>0)))
