import numpy as np
import matplotlib.pyplot as plt
import sys
import wormdatamodel as wormdm
import wormbrain as wormb
import pumpprobe as pp
import mistofrutta as mf
import os

folder = sys.argv[1]
if folder[-1]!="/": folder += "/"

save = True
stims = None
shiftvol = 0
for s in sys.argv:
    s = s.split(":")
    if s[0] == "--stim":
        if s[1]=="all" or s[1]=="unassigned":
            stims = s[1]
        else:
            try: stims = [int(s[1])] 
            except: stims = None
    if s[0] == "--shiftvol": shiftvol = int(s[1])

if stims is None: print("Specify stim.");quit()

rec = wormdm.data.recording(folder)
cervelli = wormb.Brains.from_file(folder,"brains.json")
cervelli.load_matches(folder)
info = wormb.match.load_match_parameters(folder)
ref_i = info["ref_index"]
ref_overlay, ref_labels = cervelli.get_overlay2(ref_i,return_labels=True,indices_as_labels=True)
lbls2 = cervelli.get_labels(ref_i)
for il2 in np.arange(len(lbls2)):
    if lbls2[il2] == "merge": ref_labels[il2] = "merge"
ref_labels_simple = cervelli.get_labels(ref_i)

events = rec.get_events()['optogenetics']

# Load manually identified file
if os.path.isfile(folder+"targets_manually_located.txt"):
    man_id = np.loadtxt(folder+"targets_manually_located.txt")
else:
    man_id = -1*np.ones(len(events['index']))
    
if os.path.isfile(folder+"targets_manually_located_comments.txt"):
    f = open(folder+"targets_manually_located_comments.txt","r")
    add_comments = [l[:-1] for l in f.readlines()]
    f.close()
else:
    add_comments = ["" for i in np.arange(len(events['index']))]
    
if stims=="all": 
    stims = np.arange(len(events['index']))
elif stims=="unassigned":
    stims = np.where((man_id==-1)*(np.array(add_comments)!="-1")*(np.array(add_comments)!="-2"))[0]

print("Instructions: locate the targeted neuron in the current volume and then click the corresponding neuron in the reference volume.")
for stim in stims:
    vol_i = events['index'][stim]-shiftvol
    target_coords = events['properties']['target'][stim]
    target_index = cervelli.get_closest_neuron(vol_i,target_coords,coord_ordering="xyz",z_true=True,inverse_match=False) 
    target_index_ref = cervelli.get_closest_neuron(vol_i,target_coords,coord_ordering="xyz",z_true=True,inverse_match=True) 
    print("Stim:"+str(stim))
    print("\tVolume: "+str(vol_i))
    print("\tTarget coords: "+str(target_coords))
    print("\tClosest neuron: "+str(target_index))
    print("\tClosest neuron ref: "+str(target_index_ref))
    ref_plane = np.argmin(np.abs(rec.ZZ[ref_i] - target_coords[2]))
    print("\tClosest plane in ref:"+str(ref_plane))
    cur_plane = np.argmin(np.abs(rec.ZZ[vol_i] - target_coords[2]))
    print("\tClosest plane in cur:"+str(cur_plane))
    #print("ZZ :"+str(np.array([np.arange(rec.ZZ[vol_i].shape[0]),rec.frameCount[rec.volumeFirstFrame[vol_i]:rec.volumeFirstFrame[vol_i+1]],rec.ZZ[vol_i]]).T))
    
    vol = rec.get_vol(vol_i)
    ref_vol = rec.get_vol(ref_i)
    
    _, r_c = mf.geometry.draw.crop_image(
            vol,folder,return_all=True,fname="rectangle_crop.txt",
            message="SELECT THE HEAD TO CROP THE IMAGE")
            
    overlay, labels = cervelli.get_overlay2(vol_i,return_labels=True,indices_as_labels=True)
    iperpila = mf.plt.hyperstack2(vol,overlay=overlay,overlay_labels=labels,side_views=False,plot_now=False)
    iperpila.fig.canvas.manager.set_window_title("current volume")
    iperpila.ax.plot(target_coords[0],target_coords[1],'x',c='yellow')
    iperpila.z = cur_plane
    iperpila.ax.set_ylim(r_c[0,0],r_c[1,0])
    iperpila.ax.set_xlim(r_c[1,1],r_c[0,1])
    iperpila.update()

    ref_iperpila = mf.plt.hyperstack2(ref_vol,overlay=ref_overlay,overlay_labels=ref_labels,side_views=False,plot_now=False)
    ref_iperpila.fig.canvas.manager.set_window_title("reference volume")
    ref_iperpila.z = ref_plane
    ref_iperpila.ax.set_ylim(r_c[0,0],r_c[1,0])
    ref_iperpila.ax.set_xlim(r_c[1,1],r_c[0,1])
    ref_iperpila.update()
    
    r_ipp_mngr = ref_iperpila.fig.canvas.manager
    geom = r_ipp_mngr.window.geometry()
    x,y,dx,dy = geom.getRect()
    r_ipp_mngr.window.setGeometry(dx+100, y, dx, dy)
    plt.show()

    man_id[stim] = int(ref_iperpila.get_closest_point()[1])
    print("\tManually selected neuron (in ref volume): ", man_id[stim], ref_labels[int(man_id[stim])])
    add_comment = input("\tAny additional comment\n(*int to specify a neuron, -1 to leave the automatically detected neuron, -2 for failed targeting, -3:label for neurons not found in the pp recording but identifiable in the mc image):\n")
    add_comments[stim] = add_comment
    if add_comment[:2]=="-2": 
        man_id[stim] = -2
    elif add_comment[:2]=="-1":
        man_id[stim] = -1
    elif add_comment[:2]=="-3":
        man_id[stim] = -3
    elif add_comment[:1]=="*":
        man_id[stim] = int(add_comment[1:])
    elif add_comment=="q": 
        man_id[stim] = -1
        save = input("Save the data (y/n)?")[0]=="y"
        break
    elif add_comment=="q!":
        man_id[stim] = -1
        save = False
        break
    print("\n\n")

if save:    
    header = "-1 means not manually processed, -2 failed targeting, -3:label for neurons not found in the pp recording but identifiable in the mc image"

    np.savetxt(folder+"targets_manually_located.txt",man_id.astype(int),fmt="%d",header=header)
    f = open(folder+"targets_manually_located_comments.txt","w")
    for ac in add_comments: f.write(ac+"\n")
    f.close()

