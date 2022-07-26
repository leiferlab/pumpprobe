import numpy as np, shutil, os
import pumpprobe as pp, wormdatamodel as wormdm, wormbrain as wormb

ds_tags = None 
ds_exclude_tags = "inx1"

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"               
dst_main = "/projects/LEIFER/francesco/funatlas/exported_data_full/"

# Load Funatlas for actual data
funa = pp.Funatlas.from_datasets(ds_list,load_signal=False,
                                 ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                                 verbose=False)
                                 
files_to_include = ['brains.json',
                    'fconn.pickle',
                    'green.txt',
                    'green_matchless.txt',
                    'matches.txt',
                    'matchless_nan_th.txt',
                    'recording.pickle',
                    'red.txt',
                    'red_matchless.txt',
                    'targets_manually_located.txt',
                    'targets_manually_located_comments.txt'
                    ]
                    
mc_files_to_include = ['brains.json']
                                 
for ds in np.arange(len(funa.ds_list)):
    partial_path = funa.ds_list[ds].split("/")[-2]
    dst = dst_main+partial_path+"/"
    
    # PP files    
    if not os.path.isdir(dst):
        os.mkdir(dst,mode=0o775)
        
    for f in files_to_include:
        src_f_path = funa.ds_list[ds]+f
        dst_f_path = dst+f
        if os.path.isfile(src_f_path):
            shutil.copy2(src_f_path,dst_f_path)
            
    # MC files
    cerv = wormb.Brains.from_file(funa.ds_list[ds],ref_only=True,verbose=False)
    mc_folder = cerv.labels_sources[0]
    if mc_folder is not None:
        mc_partial_path = mc_folder.split("/")[-2]
        mc_dst = dst_main+mc_partial_path+"/"
        
        if not os.path.isdir(mc_dst):
            os.mkdir(mc_dst,mode=0o775)
            
        for mc_f in mc_files_to_include:
            src_mc_f_path = mc_folder+mc_f
            dst_mc_f_path = mc_dst+mc_f
            if os.path.isfile(src_mc_f_path):
                shutil.copy2(src_mc_f_path,dst_mc_f_path)
            
    
    
    
