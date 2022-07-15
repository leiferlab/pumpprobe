import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp, wormdatamodel as wormdm, wormbrain as wormb

folder = "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20210907/pumpprobe_20210907_141336/"
signal = wormdm.signal.Signal.from_file(folder, "green")
ref_index = signal.info["ref_index"]
brains = wormb.Brains.from_file(folder,ref_only=True)
labels = np.array(brains.get_labels(0,index_for_unlabeled=True))
fconn = pp.Fconn.from_file(folder)

ie = 15

time,time2,i0,i1 = fconn.get_time_axis(ie)

ec_new = []

neu_i = 62
params = fconn.fit_params[ie][neu_i]
ec = pp.ExponentialConvolution.from_dict(params)
plt.plot(ec.eval(time2))
ec_new.append(ec.cluster_gammas_convolutional(30,ec,x=time2,atol=1e-3))

neu_i = 65
params = fconn.fit_params[ie][neu_i]
ec = pp.ExponentialConvolution.from_dict(params)
plt.plot(ec.eval(time2))
ec_new.append(ec.cluster_gammas_convolutional(30,ec,x=time2,atol=1e-3))

ecs = pp.ExponentialConvolution.acrob_cluster(3000,ec_new,x=time2,atol=1e-3)
m_gammas,fraction,f_ord = pp.ExponentialConvolution.find_matching_gammas(ecs,time=time2,return_all=True)

# I want to see if 65 is contained in 62. So I want to find the gammas that are
# not significant if convolved to 65 and then drop them from 62.
g_max = ecs[1].find_insignificant_gammas(time2,atol=1e-4)
#ecs_d=ecs[1].drop_gammas_larger_than(g_max)
print(g_max)


print(fraction)

