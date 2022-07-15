import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

folder =  sys.argv[1]
fconn = pp.Fconn.from_file(folder)

# Times on which to look for the peak time
tmax = 60
time = np.linspace(0,tmax,1000)

# Iterate over stimulations and responding neurons
peak_times = []
integrals = []
for ie in np.arange(fconn.n_stim):
    neu_is = fconn.resp_neurons_by_stim[ie]
    for i in np.arange(len(neu_is)):
        neu_i = neu_is[i]
        
        # Get the unconstrained-fit ExponentialConvolution object
        ec = fconn.get_unc_fit_ec(ie,neu_i)
        if ec is None: continue
        
        # Get the peak time and integral of the object.
        peak_time = ec.get_peak_time(time)
        integral = ec.get_integral()
        
        # Discard instances in which the time is beyond tmax, misfits.
        if peak_time<tmax:
            peak_times.append(peak_time)
            integrals.append(integral)
        
peak_times = np.array(peak_times)
integrals = np.array(integrals)

fig = plt.figure(1)     
ax = fig.add_subplot(111)
ax.plot(np.abs(integrals), peak_times,'o')

plt.show()
