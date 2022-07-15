import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

funa = pp.Funatlas()

intensity_map_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_unc31.txt")
q_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q_unc31.txt")
intensity_map_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache.txt")
q_unc31 = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q.txt")
    
#intensity_map_wt[np.isnan(intensity_map_wt)] = 0.0
intensity_map_wt[q_wt>0.05] = 0.0

#intensity_map_unc31[np.isnan(intensity_map_unc31)] = 0.0
intensity_map_unc31[q_unc31>0.05] = 0.0

d = np.abs(intensity_map_wt)-np.abs(intensity_map_unc31)
sorter = np.argsort(np.ravel(d))

ds = np.sort(np.ravel(d))
fnan = np.where(np.isnan(ds))[0][0]

i,j = sorter//q_wt.shape[0], sorter%q_wt.shape[0]

k = 0
while k<100:
    print(funa.neuron_ids[i[k]],funa.neuron_ids[j[k]],"\t",funa.neuron_ids[i[fnan-k]],funa.neuron_ids[j[fnan-k]])
    k+=1

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.imshow(intensity_map_wt-intensity_map_unc31)
plt.show()
