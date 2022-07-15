import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

rise = "--rise" in sys.argv

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
funa1 = pp.Funatlas.from_datasets(ds_list,ds_exclude_tags="unc-31 D20")
funa2 = pp.Funatlas.from_datasets(ds_list,ds_tags="unc-31")

funas = [funa1,funa2]
labels = ["wt ("+str(len(funa1.fconn))+")","unc-31 ("+str(len(funa2.fconn))+")"]

# Times on which to look for the peak time
tmax = 60
time = np.linspace(0,tmax,1000)
itmax2 = np.argmin(np.abs(time-30))
dt = time[1]-time[0]

fig = plt.figure(1,figsize=(15,8))

# Iterate over stimulations and responding neurons
peak_times = []
integrals = []
max_vals = []
ratio_to_peaks = []
decay_times = []
rise_times = []
for ifuna in np.arange(len(funas)):
    funa = funas[ifuna]
    peak_times.append([])
    integrals.append([])
    max_vals.append([])
    ratio_to_peaks.append([])
    decay_times.append([])
    rise_times.append([])
    for ifconn in np.arange(len(funa.fconn)):
        fconn = funa.fconn[ifconn]
        for ie in np.arange(fconn.n_stim):
            i0 = fconn.i0s[ie]
            i1 = 60#fconn.i1[ie]
            seg = funas[ifuna].sig[ifconn].get_segment(i0,i1,fconn.shift_vol,normalize="")
            neu_is = fconn.resp_neurons_by_stim[ie]
            for i in np.arange(len(neu_is)):
                neu_i = neu_is[i]
                
                # Get the unconstrained-fit ExponentialConvolution object
                ec = fconn.get_unc_fit_ec(ie,neu_i)
                if ec is None: continue
                
                # Get the peak time, integral, max_val, and ratio_to_peak of the 
                # object.
                peak_time = ec.get_peak_time(time)
                integral = ec.get_integral()
                integral = pp.integral(ec.eval(time),dt,8)
                
                max_val = np.max(np.abs(ec.eval(time)))
                sig = funas[ifuna].sig[ifconn][:,neu_i]
                baseline = np.median(sig[i0+fconn.shift_vol-10:i0+fconn.shift_vol])
                #stdev = np.std(sig[i0-fconn.shift_vol:i0])
                max_val /= baseline
                
                ratio_to_peak = ec.get_ratio_to_peak(time)[itmax2]
                decay_time = ec.get_effective_decay_time(time)
                rise_time = ec.get_effective_rise_time(time)
                
                peak_times[ifuna].append(peak_time)
                integrals[ifuna].append(integral)
                max_vals[ifuna].append(max_val)
                ratio_to_peaks[ifuna].append(ratio_to_peak)
                decay_times[ifuna].append(decay_time)
                rise_times[ifuna].append(rise_time)
            
    peak_times[ifuna] = np.array(peak_times[ifuna])
    integrals[ifuna] = np.array(integrals[ifuna])
    max_vals[ifuna] = np.array(max_vals[ifuna])
    ratio_to_peaks[ifuna] = np.array(ratio_to_peaks[ifuna])
    decay_times[ifuna] = np.array(decay_times[ifuna])
    rise_times[ifuna] = np.array(rise_times[ifuna])

    ax1 = fig.add_subplot(121)
    ax1.plot(np.abs(max_vals[ifuna]),decay_times[ifuna],'o',label=labels[ifuna])
    
    ax2 = fig.add_subplot(122)
    #this was when using the ratio_to_peak value
    #sel = (max_vals[ifuna]>0)*(max_vals[ifuna]<10)*(ratio_to_peaks[ifuna]<0.9)
    #sel_ratio_to_peaks = ratio_to_peaks[ifuna][sel]
    sel = (max_vals[ifuna]>0.0)*(max_vals[ifuna]<10) #max_vals>0.8
    sel_decay_times = decay_times[ifuna][sel]
    sel_rise_times = rise_times[ifuna][sel]
    sel_weights = max_vals[ifuna][sel]
    
    to_plot = sel_decay_times if not rise else sel_rise_times 
    to_plot_lbl = "decay" if not rise else "rise"
    
    #ax2.hist(sel_ratio_to_peaks,alpha=0.5,density=True,weights=sel_weights,label=labels[ifuna])
    ax2.hist(to_plot,density=True,
             range=[0,60],bins=120,weights=sel_weights,
             label=labels[ifuna],alpha=0.5)

ax1.set_xlim(0.3,10)
ax1.set_xlabel("delta F/F")
ax1.set_ylabel(to_plot_lbl+" time")
ax1.legend()
ax2.set_xlabel(to_plot_lbl+" time")
ax2.set_ylabel("density (weighted by delta F/F)")
ax2.legend()

plt.show()
