import numpy as np, sys, shutil, os
import pumpprobe as pp, wormdatamodel as wormdm, wormbrain as wormb
#wormdatamodel is not necessary but its import will serve as a check that all
#the libraries are installed

print("\n\n\nPlease be patient, this script might take some time because it has to rewrite files.\n\n\n")

your_folder = sys.argv[1]
if your_folder[-1] != "/": your_folder+="/"

ds_list = your_folder+"funatlas_list_orig.txt"

f = open(ds_list,"r")
lines = f.readlines()
f.close()

folders = []
for il in np.arange(len(lines)):
    line = lines[il]
    orig_folder, tags = line.split("#")
    new_folder = your_folder+orig_folder
    
    # store as new folders
    folders.append(new_folder)
    
    # remake string to save in new ds_list file
    lines[il] = new_folder+"#"+tags
    
# save to new ds_list file
f = open(your_folder+"funatlas_list.txt","w")
for il in np.arange(len(lines)):
    f.write(lines[il])
f.close()

# files_containing_paths = ['brains.json','fconn.pickle']
# plus the multicolor brains

for ds in np.arange(len(folders)):
    # Get reference volume index
    match_info = wormb.match.load_match_parameters(folder[ds])
    ref_index = match_info["ref_index"]
    
    # Update folder paths in brains, both its own path and the path of the 
    # linked multicolor recording.
    cerv = wormb.Brains.from_file(folders[ds])
    cerv.folder = folders[ds]
    
    mc_cerv_folder = cerv.labels_sources[ref_index]
    mc_cerv_folder_partial = mc_cerv_folder.split("/")[-2]
    # just a check
    if mc_cerv_folder_partial == "": 
        mc_cerv_folder_partial = mc_cerv_folder.split("/")[-1]
    mc_cerv_folder = your_folder+mc_cerv_folder_partial+"/"
    cerv.labels_sources[ref_index] = mc_cerv_folder
    
    # Save the brains file    
    cerv.to_file(folders[ds])
    
    # Update the path inside the multicolor brains file.
    mc_cerv = wormb.Brains.from_file(mc_cerv_folder)
    mc_cerv.folder = mc_cerv_folder
    mc_cerv.to_file(mc_cerv_folder)
    
    # Update the folder inside the fconn file.
    fconn = pp.Fconn.from_file(folders[ds])
    fconn.folder = folders[ds]
    fconn.to_file(folders[ds])
    
    
    
