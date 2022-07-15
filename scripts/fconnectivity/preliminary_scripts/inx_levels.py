import numpy as np
import pumpprobe as pp

funa = pp.Funatlas(merge_bilateral=True,merge_dorsoventral=False,merge_numbered=False)

inx1 = np.around(funa.get_inx_exp_levels2("inx-1"),2)
inx7 = np.around(funa.get_inx_exp_levels2("inx-7"),2)

for ai in np.arange(funa.n_neurons):
    print(funa.neuron_ids[ai],inx1[ai],inx7[ai])
