import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

funa = pp.Funatlas()
funa2 = pp.Funatlas(merge_bilateral=True)

intensity_map_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_unc31.txt")
q_wt = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q_unc31.txt")

intensity_map_wt[q_wt>0.05] = 0.0

act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_bilateral_merged.txt")
act_conn[np.isnan(act_conn)] = 0

int_map2 = np.zeros((funa2.n_neurons,funa2.n_neurons))
count = np.zeros((funa2.n_neurons,funa2.n_neurons))
for i in np.arange(intensity_map_wt.shape[0]):
    for j in np.arange(intensity_map_wt.shape[1]):
        ai2 = funa2.ids_to_i([funa.neuron_ids[i]])
        aj2 = funa2.ids_to_i([funa.neuron_ids[j]])
        
        if not np.isnan(intensity_map_wt[i,j]):
            int_map2[ai2,aj2] += intensity_map_wt[i,j]
            count[ai2,aj2] += 1
        
int_map2[count!=0] /= count[count!=0]

d = np.abs(int_map2)-np.abs(act_conn)
sorter = np.argsort(np.ravel(d))[::-1]

ds = np.sort(np.ravel(d))

i,j = sorter//int_map2.shape[0], sorter%int_map2.shape[0]

k = 0
while k<100:
    print(funa2.neuron_ids[i[k]],funa2.neuron_ids[j[k]])#,int_map2[i[k],j[k]],act_conn[i[k],j[k]])
    k+=1

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.imshow(int_map2-act_conn)
plt.show()
