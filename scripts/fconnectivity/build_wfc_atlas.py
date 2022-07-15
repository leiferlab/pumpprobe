import numpy as np, os, pickle, sys
import wormfunconn as wfc, pumpprobe as pp
import matplotlib.pyplot as plt

unc31 = "--unc31" in sys.argv
strain = "" if not unc31 else "unc31"
ds_tags = None if not unc31 else "unc31"
ds_exclude_tags = "mutant" if not unc31 else None

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
                  
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": "sg_causal", 
                 "smooth_n": 13, "smooth_poly": 1,
                 "photobl_appl":True,            
                 "matchless_nan_th_from_file": True}
                 

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,merge_bilateral=False,signal="green",
                                 signal_kwargs = signal_kwargs,
                                 enforce_stim_crosscheck=True,
                                 ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,)

occ1,occ2 = funa.get_occurrence_matrix(req_auto_response=True)
occ3 = funa.get_observation_matrix(req_auto_response=True)

if unc31:
    intensity_map = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_unc31.txt")
    q = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q_unc31.txt")
else:
    intensity_map = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache.txt")
    q = np.loadtxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_cache_q.txt")
    
km = funa.get_kernels_map_ec(occ2,occ3,drop_saturation_branches=False,include_flat_kernels=False)

flat_kernel = wfc.ExponentialConvolution_min([{"g":1.0,"factor":0.0,"power_t":0,"branch":0}])

intensity_map[q>0.05] = np.nan
np.fill_diagonal(intensity_map,0.0)
intensity_map = funa.reduce_to_head(intensity_map)

neu_ids = np.copy(funa.head_ids)
km[q>0.05] = flat_kernel
km = funa.reduce_to_head(km)
km_min = np.empty_like(km)

for i in np.arange(km.shape[0]):
    for j in np.arange(km.shape[1]):
        if km[i,j] is not None:
            km_min[i,j] = wfc.ExponentialConvolution_min(km[i,j].exp[-1])
        else:
            km_min[i,j] = None

dst_folder = os.path.join("/home/frandi/dev/worm-functional-connectivity/atlas/")
if strain == "": 
    strain = "wild-type"
elif strain == "unc31":
    strain = "unc-31"

fa = wfc.FunctionalAtlas(np.copy(neu_ids),km_min)
fa.scalar_atlas = np.copy(intensity_map)

dst_folder = os.path.join("/home/frandi/dev/worm-functional-connectivity/atlas/")
if strain == "": 
    strain = "wild-type"
elif strain == "unc31":
    strain = "unc-31"
fa.to_file(dst_folder,strain)
