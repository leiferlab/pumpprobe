import numpy as np, sys, matplotlib.pyplot as plt
import pumpprobe as pp, wormdatamodel as wormdm, wormbrain as wormb

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
#ds_list = ["/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210907/pumpprobe_20210907_141336/",
#           #"/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210903/pumpprobe_20210903_153005/",
#           "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210907/pumpprobe_20210907_110334/"]

g_th = 1.
max_hops = 1
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--g-th": g_th = float(sa[1])
    if sa[0] == "--max-hops": 
        if sa[1] in ["none","None"]:
            max_hops = None
        else:
            max_hops = int(sa[1])

funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=True,merge_dorsoventral=False,merge_AWC=True,ds_tags="",ds_exclude_tags="old mutant")

# Initialize various matrices 
has_fast_g = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
has_any_conn = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
fastest_g = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
fastest_g_ph_oth = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
fastest_g_ph_ph = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
fastest_g_oth_oth = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
fastest_g_oth_ph = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan

peak_time = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
peak_time_ph_oth = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
peak_time_ph_ph = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
peak_time_oth_oth = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan
peak_time_oth_ph = np.ones((len(funa.ds_list),funa.n_neurons,funa.n_neurons))*np.nan

time = np.linspace(0,50,500)


kernels_pharynx = []

# Iterate over datasets and stimulation, but then separately store results for 
# reponding neurons in the pharyngeal network.
for ds in np.arange(len(funa.ds_list)):
    for ie in np.arange(funa.fconn[ds].n_stim):
        neu_aj = funa.stim_neurons_ai[ds][ie]
        neu_is = funa.fconn[ds].resp_neurons_by_stim[ie]
        
        # Skip if the stimulated neuron did not respond. Although this control
        # is probably not necessary, as I'm already checking that ec is not
        # None below.
        if funa.fconn[ds].stim_neurons[ie] not in neu_is:
            continue
            
        # Set to zero the entries for this stimulated neuron and the neurons
        # observed in the recording, unless you already did it before.
        if not neu_aj in funa.stim_neurons_ai[ds][:ie]:
            has_any_conn[ds,funa.atlas_i[ds],neu_aj] = 0
        
        for i in np.arange(len(neu_is)):
            neu_i = neu_is[i]
            neu_ai = funa.atlas_i[ds][neu_i]
            
            has_any_conn[ds,neu_ai,neu_aj] = 1
            
            # Get the ExponentialConvolution for the neu_i,neu_j
            # (using the ds_indices)
            ec = funa.fconn[ds].get_kernel_ec(ie,neu_i)
            if ec is None: continue
            #ec.cluster_gammas_convolutional(10,ec,x=time,atol=5e-2)
            
            # Get whether the kernel has a fast gamma.
            has_fast_g_ = ec.has_gamma_faster_than(g_th)
            # Get the fastest gamma
            #fastest_g_ = np.max(ec.get_bare_params()[0])
            fastest_g_ = np.max(ec.get_branches_ampl_avggamma()[1])
            # Get the peak time of the kernel
            peak_time_ = ec.get_peak_time(time)
            
            # Store it with a logical or, so that you don't risk overwriting
            # previous positive cases.
            if not has_fast_g[ds,neu_ai,neu_aj]>0:
                # it's either 0 or nan
                has_fast_g[ds,neu_ai,neu_aj] = has_fast_g_
            
            # Store fastest gamma and peak time for any connection
            fastest_g[ds,neu_ai,neu_aj] = np.nanmax([fastest_g[ds,neu_ai,neu_aj],
                                                 fastest_g_])
            peak_time[ds,neu_ai,neu_aj] = np.nanmax([peak_time[ds,neu_ai,neu_aj],
                                                 peak_time_])
            
            # Store fastest gamma and peak time for the connections from
            # outside the pharyngeal network to the pharyngeal network.
            if neu_ai in funa.pharynx_ai and neu_aj not in funa.pharynx_ai: 
                fastest_g_ph_oth[ds,neu_ai,neu_aj] = np.nanmax([fastest_g_ph_oth[ds,neu_ai,neu_aj],
                                                 fastest_g_])
                peak_time_ph_oth[ds,neu_ai,neu_aj] = np.nanmax([peak_time_ph_oth[ds,neu_ai,neu_aj],
                                                 peak_time_])
                
                kernels_pharynx.append(ec.eval(time))
            elif neu_ai in funa.pharynx_ai and neu_aj in funa.pharynx_ai:
                fastest_g_ph_ph[ds,neu_ai,neu_aj] = np.nanmax([fastest_g_ph_ph[ds,neu_ai,neu_aj],
                                                 fastest_g_])
                peak_time_ph_ph[ds,neu_ai,neu_aj] = np.nanmax([peak_time_ph_ph[ds,neu_ai,neu_aj],
                                                 peak_time_])
            elif neu_ai not in funa.pharynx_ai and neu_aj not in funa.pharynx_ai:
                fastest_g_oth_oth[ds,neu_ai,neu_aj] = np.nanmax([fastest_g_oth_oth[ds,neu_ai,neu_aj],
                                                 fastest_g_])
                peak_time_oth_oth[ds,neu_ai,neu_aj] = np.nanmax([peak_time_oth_oth[ds,neu_ai,neu_aj],
                                                 peak_time_])
            elif neu_ai not in funa.pharynx_ai and neu_aj in funa.pharynx_ai:
                fastest_g_oth_ph[ds,neu_ai,neu_aj] = np.nanmax([fastest_g_oth_ph[ds,neu_ai,neu_aj],
                                                 fastest_g_])
                peak_time_oth_ph[ds,neu_ai,neu_aj] = np.nanmax([peak_time_oth_ph[ds,neu_ai,neu_aj],
                                                 peak_time_])
                                                 

# Select the entries in which has_fast_g is not nan (i.e. the pair was observed)
'''has_fast_g_nanmask = np.all(np.isnan(has_fast_g),axis=0)
has_fast_g2 = np.clip(np.nansum(has_fast_g,axis=0),0,1)
has_fast_g2 = np.ravel(has_fast_g2[~has_fast_g_nanmask])'''

def meannanclip(array):
    array2 = np.nanmean(array,axis=0)
    mask = np.isnan(array2)
    array2 = np.clip(np.ravel(array2[~mask]),0,2)
    return array2
    
def meannan(array):
    array2 = np.nanmean(array,axis=0)
    mask = np.isnan(array2)
    array2 = np.ravel(array2[~mask])
    return array2
'''
fastest_g2 = np.nanmean(fastest_g,axis=0)
nanmask1 = np.isnan(fastest_g2)
fastest_g2 = np.clip(np.ravel(fastest_g2[~nanmask1]),0,2) # Clip them so that you capture all the large gammas

fastest_g_ph_oth2 = np.nanmean(fastest_g_ph_oth,axis=0)
nanmask2 = np.isnan(fastest_g_ph_oth2)
fastest_g_ph_oth2 = np.clip(np.ravel(fastest_g_ph_oth2[~nanmask2]),0,2) # Clip them so that you capture all the large gammas

peak_time = np.nanmean(peak_time,axis=0)
peak_time = np.ravel(peak_time[~has_fast_g_nanmask])
peak_time_ph_oth = np.nanmean(peak_time_ph_oth,axis=0)
peak_time_ph_oth = np.ravel(peak_time_ph_oth[~has_fast_g_nanmask])
'''

fastest_g = meannanclip(fastest_g)
fastest_g_ph_oth = meannanclip(fastest_g_ph_oth)
fastest_g_ph_ph = meannanclip(fastest_g_ph_ph)
fastest_g_oth_oth = meannanclip(fastest_g_oth_oth)
fastest_g_oth_ph = meannanclip(fastest_g_oth_ph)

peak_time = meannan(peak_time)
peak_time_ph_oth = meannan(peak_time_ph_oth)
peak_time_ph_ph = meannan(peak_time_ph_ph)
peak_time_oth_oth = meannan(peak_time_oth_oth)
peak_time_oth_ph = meannan(peak_time_oth_ph)

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.hist(fastest_g_oth_oth,bins=30,density=True,range=(0,2),label="fastest g !pharynx->!pharynx ("+str(len(fastest_g_oth_oth))+" kernels)",alpha=0.5)
ax1.hist(fastest_g_ph_oth,bins=30,density=True,range=(0,2),label="fastest g !pharynx->pharynx ("+str(len(fastest_g_ph_oth))+" kernels)",alpha=0.5)
ax1.set_xlabel("g (s^-1)")
ax1.set_title("fastest gamma in kernel")
ax1.legend()

fig = plt.figure(2)
ax1 = fig.add_subplot(111)
#ax1.hist(peak_time,bins=30,density=True,label="peak time",alpha=0.5)
ax1.hist(peak_time_oth_oth,bins=30,density=True,label="peak time !pharynx->!pharynx",alpha=0.5)
ax1.hist(peak_time_ph_oth,bins=30,density=True,label="peak time !pharynx->pharynx",alpha=0.5)
ax1.set_xlabel("t* (s)")
ax1.set_title("peak time of kernel")
ax1.legend()

fig = plt.figure(3)
ax1 = fig.add_subplot(111)
ax1.hist(fastest_g_oth_oth,bins=30,density=True,range=(0,2),label="fastest g !pharynx->!pharynx ("+str(len(fastest_g_oth_oth))+" kernels)",alpha=0.5)
ax1.hist(fastest_g_ph_ph,bins=30,density=True,range=(0,2),label="fastest g pharynx->pharynx ("+str(len(fastest_g_ph_ph))+" kernels)",alpha=0.5)
ax1.set_xlabel("g (s^-1)")
ax1.set_title("fastest gamma in kernel")
ax1.legend()

fig = plt.figure(4)
ax1 = fig.add_subplot(111)
ax1.hist(peak_time_oth_oth,bins=30,density=True,label="peak time !pharynx->!pharynx",alpha=0.5)
ax1.hist(peak_time_ph_ph,bins=30,density=True,label="peak time pharynx->pharynx",alpha=0.5)
ax1.set_xlabel("t* (s)")
ax1.set_title("peak time of kernel")
ax1.legend()

fig = plt.figure(5)
ax1 = fig.add_subplot(111)
ax1.hist(fastest_g_oth_oth,bins=30,density=True,range=(0,2),label="fastest g !pharynx->!pharynx ("+str(len(fastest_g_oth_oth))+" kernels)",alpha=0.5)
ax1.hist(fastest_g_oth_ph,bins=30,density=True,range=(0,2),label="fastest g pharynx->!pharynx ("+str(len(fastest_g_oth_ph))+" kernels)",alpha=0.5)
ax1.set_xlabel("g (s^-1)")
ax1.set_title("fastest gamma in kernel")
ax1.legend()

fig = plt.figure(6)
ax1 = fig.add_subplot(111)
ax1.hist(peak_time_oth_oth,bins=30,density=True,label="peak time !pharynx->!pharynx",alpha=0.5)
ax1.hist(peak_time_ph_oth,bins=30,density=True,label="peak time pharynx->!pharynx",alpha=0.5)
ax1.set_xlabel("t* (s)")
ax1.set_title("peak time of kernel")
ax1.legend()

'''fig = plt.figure(3)
ax1 = fig.add_subplot(111)
for k in kernels_pharynx:
    ax1.plot(time,k)'''

plt.show()


