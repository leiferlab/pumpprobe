import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm
import pumpprobe as pp, wormdatamodel as wormdm

folder = "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210907/pumpprobe_20210907_141336/" 
fconn = pp.Fconn.from_file(folder)
signal = wormdm.signal.Signal.from_file(folder, "green")

ie = 15
time,time2,i0,i1 = fconn.get_time_axis(ie)
time3 = np.linspace(0,time2[-1],1000)

neu_j = fconn.stim_neurons[ie]

y_stim_sig = signal.get_segment(i0,i1,fconn.shift_vol)[:,neu_j]

params = fconn.fit_params_unc[ie][neu_j]
ec = pp.ExponentialConvolution.from_dict(params)

y_stim = ec.eval(time3)

n_gammas = 10
c = cm.viridis(np.linspace(0,1,n_gammas))
gammas = np.linspace(0.1,3,n_gammas)

fig = plt.figure(1)
ax = fig.add_subplot(111)
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax.plot(time3,y_stim,c="k",lw=8,label="stim. fit")
ax.plot(time,y_stim_sig,c="r",label="stim. data")

for i in np.arange(n_gammas):
    gamma = gammas[i]
    f = gamma**3/2*time3**2*np.exp(-gamma*time3)
    #f+= -0.9*(gamma*1.1)**3/2*time3**2*np.exp(-gamma*1.1*time3)
    y = pp.convolution(y_stim,f,time3[1]-time3[0],8)
    
    ax.plot(time3,y,label=str(np.around(gamma,2)),c=c[i])
    ax2.plot(time3,f,label=str(np.around(gamma,2)),c=c[i])

ax.set_xlabel("Time (s)")    
ax.legend(loc=2)
plt.show()
