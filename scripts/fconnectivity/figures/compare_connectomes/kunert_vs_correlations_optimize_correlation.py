import numpy as np, matplotlib.pyplot as plt
import os, sys, json
import pumpprobe as pp

folder = "/projects/LEIFER/francesco/simulations/activity_connectome_sign2/"
print("using",folder)

funa = pp.Funatlas(merge_bilateral=False,merge_dorsoventral=False,
                   merge_numbered=False,merge_AWC=False)
                   
funa2 = pp.Funatlas(merge_bilateral=True,merge_dorsoventral=False,
                   merge_numbered=False,merge_AWC=True)

corr = np.zeros((funa2.n_neurons,funa2.n_neurons,funa2.n_neurons))*np.nan
count = np.zeros((funa2.n_neurons,funa2.n_neurons,funa2.n_neurons))

for neu_j in np.arange(funa.n_neurons):
    print(neu_j,end="")
    neu_id = funa.neuron_ids[neu_j]
    
    y = np.loadtxt(folder+neu_id+".txt")
    
    #Get the ai in the merged funatlas
    neu_j2 = funa2.ids_to_i([neu_id])
    
    y = y[:y.shape[0]//2]
    y -= y[:,0][:,None]
    
    r = np.corrcoef(y)
    
    for neu_i in np.arange(funa.n_neurons):
        #Get the ai in the merged funatlas
        neu_i2 = funa2.ids_to_i([funa.neuron_ids[neu_i]])
        for neu_k in np.arange(funa.n_neurons):
            neu_k2 = funa2.ids_to_i([funa.neuron_ids[neu_k]])
            if np.isnan(corr[neu_j2,neu_i2,neu_k2]):
                corr[neu_j2,neu_i2,neu_k2] = r[neu_i,neu_k]
                count[neu_j2,neu_i2,neu_k2] += 1
            else:
                corr[neu_j2,neu_i2,neu_k2] += r[neu_i,neu_k]
                count[neu_j2,neu_i2,neu_k2] += 1
    print("\r",end="")

corr = corr/count

np.save(folder+"activity_connectome_correlation_individual.npy",corr)
