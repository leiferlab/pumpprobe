import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

drop_saturation_branches = True

funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="mutant",#None"mutant",
                verbose=False)
                
occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=True)

time1 = np.linspace(0,30,1000)
time2 = np.linspace(0,10,1000)

rise_times, kernels = funa.get_eff_rise_times(
                            occ2,time2,True,drop_saturation_branches,
                            return_kernels=True)

occ2b,_ = funa.filter_occ2(occ2,rise_times,leq=0.55)
kernelsb, _ = funa.filter_occ2(kernels,rise_times,leq=0.55)
kernelsb_rav = funa.ravel_occ2(kernelsb)

n_neu = funa.n_neurons
autoconv = np.empty((n_neu,n_neu),dtype=object)

ac_rt = []
time = np.linspace(0,2,1000)
dt = time[1]-time[0]
oneovere = np.exp(-1)
print(len(kernelsb_rav))
for i in np.arange(len(kernelsb_rav)):
    if i%10==0: print(i/len(kernelsb_rav))
    for j in np.arange(len(kernelsb_rav)):
        ker1 = kernelsb_rav[i]
        ker2 = kernelsb_rav[j]
        if ker1 is None or ker2 is None: continue
        
        y1 = ker1.eval(time)
        y2 = ker2.eval(time)
        
        y = pp.convolution(y1,y2,dt,8)
        y = np.abs(y)
        
        # THIS ASSUMES YOU'VE DROPPED THE SATURATION BRANCHES
        # TODO TODO TODO
        yargm = np.argmax(y)
        ym = y[yargm]
        rt = yargm-np.argmin(np.abs(y[:yargm]-ym*oneovere))
        
        ac_rt.append(rt*dt)

ac_rt = np.array(ac_rt)
np.savetxt("/home/frandi/autoconvolution_risetimes.txt",ac_rt)

fig = plt.figure(1)
ax = fig.add_subplot(111)
rise_times_rav = funa.ravel_occ2(rise_times)
ax.hist(rise_times_rav,bins=100,range=(0,2),alpha=0.5,density=True)
ax.hist(rise_times_rav,bins=100,range=(0,0.55),alpha=0.5,density=True)
ax.hist(ac_rt,bins=100,range=(0,2),alpha=0.5,density=True)
ax.set_xlabel("rise time (s)")
plt.savefig("/home/frandi/autoconvolution_risetimes.png",dpi=300,bbox_inches="tight")
plt.show()
