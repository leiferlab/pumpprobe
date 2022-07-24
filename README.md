# pumpprobe

This repository contains code for the high-level analysis of the functional connectivity data in Randi et al. (submitted). It contains several scripts to generate figures, that rely on the 3 main classes Funatlas, Fconn, and ExponentialConvolution, as well as the Python modules [wormdatamodel](github.com/leiferlab/wormdatamodel) and [wormbrain](github.com/leiferlab/wormbrain). The figures of the paper are reproduced using the commands found in the scripts/fconnectivity/figures/paper/ folder.

### Funatlas
Funatlas is the functional atlas class that aggregates functional connectivity data from multiple datasets from different animals. The class can be instantiated via the regular constructor or from the class method from_datasets(), which will load the data from several datasets and compile the results based on neural identities. Funatlas can be instantiated to maintain the "exact" neural identities, or to merge neurons into classes (i.e. to approximate neuron identities): `merge_bilateral=True` will merge results for, e.g., AVAL and AVAR into the class AVA_, `merge_dorsoventral=True` will merge RMED and RMEV into RME_, while `merge_numbered=True` will merge VB3, VB4, ... into VB. These options can be combined to merge, for example, SMBVL, SMBVR, SMBDL, and SMBDR into SMB__ with `merge_bilateral=True, merge_dorsoventral=True`. 
Important methods regarding neuron identities are

* Funatlas.ids_to_ai() converts neuron identities to atlas-indices (ai), given the neuron-class-merging options with which the object has been created.

* Funatlas.i_to_ai() converts dataset-specific indices of neurons into atlas-indices. For example, in a given dataset, neuron ADAL could be neuron number 38, but in the atlas list of IDs, ADAL is number 0 (i.e. its atlas-index is 0). 

The main goal of the Funatlas is to aggregate data from many datasets, for example obtaining all the responses of neuron AVER to stimulations of neuron AVDL. When Funatlas is created using Funatlas.from_datasets(), you can obtain the matrices occ1, occ2, and occ3 using the functions

* occ1, occ2 = Funatlas.get_occurrence_matrix()
* occ3 = Funatlas.get_observation_matrix()

`occ1[i,j]` is the number of times neuron i responded to stimulations of neuron j, `occ3[i,j]` is the total number of times one could have observed a response of i to stimulation of j (regardless of whether there was a response or not). `occ2[i,j]` is a list of dictionaries that contain the relevant information to retrieve the neural activity traces of the responses of i to stimulations of j.

Many other methods return aggregated results from the datasets, like `Funatlas.get_deltaFoverF()`, `Funatlas.get_eff_rise_times()`, and `Funatlas.get_signal_correlations()`, in addition to many more, including utilities functions.

Finally, Funatlas also has methods to load and use the _C. elegans_ anatomical connectome (from White et al. 1986 and Witvliet et al. 2020), the known extrasynaptic connectome (from Bentley et al. 2016), and gene expression data from CeNGEN (Taylor et al. 2021). 
