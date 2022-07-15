import numpy as np
import wormbrain as wormb
import wormdatamodel as wormdm
import pumpprobe as pp
import unmix as um
import mistofrutta as mf
import matplotlib.pyplot as plt
import sys
import os
import pickle

tubatura = pp.Pipeline("match_pp_multicolor.py",sys.argv)

reg = tubatura.config['reg_params']['reg_type']
match_method = tubatura.config['match_method']
beta = tubatura.config['reg_params']['beta']
llambda = tubatura.config['reg_params']['lambda']
neighbor_cutoff = tubatura.config['reg_params']['neighbor_cutoff']
dsmm_max_iter = 100000#tubatura.config['reg_params']['max_iter']
gamma0 = 3.0#10.
eq_tol = 10e-5
conv_epsilon = 10e-5
fullpy = False
mc_resize_ratio = 2
for a in sys.argv:
    sa = a.split(":")
    if sa[0] == "--multicolor" or sa[0] == "--mc": folder_mc = sa[1]
    if sa[0] == "--pp": folder_pp = sa[1]
    if sa[0] == "--mc-resize-ratio": mc_resize_ratio = int(sa[1])
    
# Set terminal window title to be able to control it
term_title = "mcpp_"+"_".join(folder_pp.split("/")[-2].split("_")[1:])
print("\x1b]2;"+term_title+"\x07",end="\r")
    
cerv_mc = wormb.Brains.from_file(folder_mc)
cerv_pp = wormb.Brains.from_file(folder_pp)

ref_vol_i = wormb.match.load_match_parameters(folder_pp)["ref_index"]
print("Reference volume index :", ref_vol_i)
print("The direction of the reference volume is ",np.sign(cerv_pp.zOfFrame[ref_vol_i][-1]-cerv_pp.zOfFrame[ref_vol_i][0]))

pts_mc = cerv_mc.trueCoords(vol=0,coord_ordering="xyz").copy()
pts_mc[:,:2] *= mc_resize_ratio
pts_ref = cerv_pp.trueCoords(vol=ref_vol_i,coord_ordering="xyz").copy()

#cerv_pp.plot(0,mode="2d",plotNow=False)
#cerv_mc.plot(0,mode="2d")

im = um.load_spinningdisk_uint16(folder_mc)

if not os.path.isfile(folder_mc+'polygon.pickle'):
    poly = mf.geometry.draw.polygon(np.sum(im[:,2],axis=0))
    polygon = poly.getPolygon()
    pickle_file = open(folder_mc+'polygon.pickle',"wb")
    pickle.dump(polygon,pickle_file)
    pickle_file.close()
else:
    pickle_file = open(folder_mc+'polygon.pickle',"rb")
    polygon = pickle.load(pickle_file)
    pickle_file.close()
pts_mc_sel, poly_mask = mf.geometry.draw.select_points(pts_mc, polygon, method='polygon',return_all=True)
pts_mc = np.copy(pts_mc_sel)
'''
im_mc, n_z, n_ch = um.load_spinningdisk_uint16(folder_mc,return_all=True)
line_mc = mf.geometry.draw.line(np.sum(im_mc[:,-2],axis=0),plot_now=False)
rec = wormdm.data.recording(folder_pp)
rec.load(startVolume=ref_vol_i, nVolume=1)
line_pp = mf.geometry.draw.line(np.sum(rec.frame[:,0],axis=0))

pt_mc = line_mc.get_line()[0]
pt_pp = line_pp.get_line()[0]
print(pt_pp)
print(pt_mc)

pts_mc2 = np.copy(pts_mc)
pts_ref2 = np.copy(pts_ref)
pts_mc2[:,1:] -= pt_mc[::-1]
pts_ref2[:,1:] -= pt_pp[::-1]

pts_mc2[:,0] /= np.max(pts_mc2[:,0])-np.min(pts_mc2[:,0])
pts_mc2[:,1] /= np.max(pts_mc2[:,0])-np.min(pts_mc2[:,1])
pts_mc2[:,2] /= np.max(pts_mc2[:,0])-np.min(pts_mc2[:,2])

pts_ref2[:,0] /= np.max(pts_ref2[:,0])-np.min(pts_ref2[:,0])
pts_ref2[:,1] /= np.max(pts_ref2[:,0])-np.min(pts_ref2[:,1])
pts_ref2[:,2] /= np.max(pts_ref2[:,0])-np.min(pts_ref2[:,2])
'''
'''Match = wormb.match.match(
            pts_mc, pts_ref, 
            method=match_method, registration=reg, beta=3.0, llambda=2.,
            gamma0=gamma0, neighbor_cutoff=neighbor_cutoff,
            eq_tol=eq_tol, conv_epsilon=conv_epsilon,# max_iter=dsmm_max_iter,
            fullpy=fullpy)'''
#Match = wormb.match.match(pts_mc2, pts_ref2, method="nearest", distanceThreshold=0.0)#2.0
#Match[Match<0] = -1
'''
labels = np.array(cerv_mc.get_labels(0))[poly_mask][Match]
labels[Match==-1] = "-1"
original_indexes_match = np.arange(len(poly_mask))[poly_mask][Match]
print(original_indexes_match)
labels_match = []
for i in np.arange(len(labels)):
    if labels[i]!="": labels_match.append(labels[i])
    else: labels_match.append(Match[i])
labels_match = np.array(labels_match,dtype="<U5")
print(labels_match)
#wormb.match.plot_matches(pts_mc2, pts_ref2, Match, mode="2d", showAll=False)
'''
#fix_matching = input("Fix matching (y/n)?")[0] == "y"
fix_matching = True
if fix_matching: 
    print("Fixing the matches")
    # Load multicolor images
    im_mc, n_z, n_ch = um.load_spinningdisk_uint16(folder_mc,return_all=True)
    manual_labels_mc = [cerv_mc.get_labels(0)]
    ovrl_mc, ovrl_labs_mc = cerv_mc.get_overlay2(vol=0,return_labels=True,label_size=5,scale=2,indices_as_labels=True,index_for_unlabeled=False)
    im_mc, r_c = mf.geometry.draw.crop_image(im,folder_mc,return_all=True,scale=2,fname="rectangle_crop.txt")
    ovrl_mc[:,1] -= r_c[0,0]; ovrl_mc[:,2] -= r_c[0,1]
    ipp_mc = mf.plt.hyperstack2(im_mc[:,2],overlay=ovrl_mc,overlay_labels=ovrl_labs_mc,side_views=False,plot_now=False)
    ipp_mc.fig.canvas.set_window_title("mc")
    ipp_mc.overlay_fontsize = 10
    
    # Load pumpprobe images
    rec = wormdm.data.recording(folder_pp)
    rec.load(startVolume=ref_vol_i, nVolume=1)
    ovrl_pp, ovrl_labs_pp = cerv_pp.get_overlay2(vol=ref_vol_i,return_labels=True,label_size=5,index_for_unlabeled=False,lookup_source=False)
    manual_labels_tmp = [cerv_pp.get_labels(ref_vol_i,lookup_source=False)]
    ipp_pp = mf.plt.hyperstack2(rec.frame[:,0],overlay=ovrl_pp,overlay_labels=ovrl_labs_pp,manual_labels=manual_labels_tmp,side_views=False,plot_now=False)
    ipp_pp.fig.canvas.set_window_title("pp")
    ipp_pp.overlay_fontsize = 10
    ipp_pp.parent_term = term_title
plt.show()    

save = input("Save results? This will overwrite previously saved data. (y/n)")[0] == "y"
if save:
    new_manual_labels = np.array(ipp_pp.get_manual_labels()[0],dtype="<U5")
    print(new_manual_labels)
    cerv_pp.set_labels(ref_vol_i,new_manual_labels)
    cerv_pp.labels_sources[ref_vol_i] = folder_mc
    cerv_pp.to_file(folder_pp)
else:
    print("Results not saved.")
            

#wormb.match.plot_matches(pts_mc, pts_ref, Match, mode="3d", showAll=False)
