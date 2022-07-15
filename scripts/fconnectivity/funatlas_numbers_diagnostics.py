import numpy as np, sys
import pumpprobe as pp

req_folder = None
req_ds = None
ds_tags = None
ds_exclude_tags = None
enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
fname = "/projects/LEIFER/francesco/funatlas_list.txt"
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--ds-list": fname=sa[1]
    if sa[0] == "--folder": req_folder = sa[1]
    if sa[0] == "--ds": req_ds = int(sa[1])
    if sa[0] == "--ds-tags": ds_tags = " ".join(sa[1].split("-"))
    if sa[0] == "--ds-exclude-tags": ds_exclude_tags = " ".join(sa[1].split("-"))

funa = pp.Funatlas.from_datasets(
                fname,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,load_signal=False,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck)
                
n_ds = len(funa.ds_list)

occ1, _ = funa.get_occurrence_matrix(req_auto_response = False)

n_neu1 = np.sum( np.sum(occ1,axis=0)!=0 )
n_neu2 = np.sum( np.sum(occ1,axis=1)!=0 )

n_neu = [ len(funa.atlas_i[i]) for i in np.arange(n_ds)]
n_labels_neu = [ np.sum(funa.atlas_i[i]>=0) for i in np.arange(n_ds)]

n_successful_stims = [ np.sum(funa.fconn[i].stim_neurons >=0) for i in np.arange(n_ds)]
n_successful_labeled_stims = [ np.sum(funa.stim_neurons_ai[i] >=0) for i in np.arange(n_ds)]
n_stims = [len(funa.fconn[i].stim_neurons) for i in np.arange(n_ds)]


for i in np.arange(n_ds):
    print(i,funa.ds_list[i])
    print("\tNumber of neurons:",n_neu[i])
    print("\tNumber of labeled neurons:",n_labels_neu[i])
    print("\tNumbers of stimulations:",n_stims[i])
    print("\tNumbers of successful stimulations:",n_successful_stims[i])
    print("\tNumbers of successful labeled stimulations:",n_successful_labeled_stims[i])

if req_folder is not None:
    print(req_folder,"is ds",funa.ds_list.index(req_folder))
if req_ds is not None:
    print("ds",req_ds,"is",funa.ds_list[req_ds])

print("\n##Global statistics\n")
print("Number of datasets:",n_ds)
print("Number of resp neuron classes:",n_neu2)
print("Number of stim neuron classes:",n_neu1)
print("Number of labeled neurons:",n_labels_neu)
print("Average number of labeled neurons:",int(np.average(n_labels_neu)))
print("Numbers of stimulations:",n_stims)
print("Numbers of successful stimulations:",n_successful_stims)
print("Numbers of successful labeled stimulations:",n_successful_labeled_stims)
print("Total number of stimulations:", np.sum(n_stims))
print("Total number of successful stimulations:",np.sum(n_successful_stims))
print("Total number of successful labeled stimulations:",np.sum(n_successful_labeled_stims))
