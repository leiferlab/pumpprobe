import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wormdatamodel as wormdm
import wormbrain as wormb
import pumpprobe as pp
import mistofrutta as mf
import sys
import os
import subprocess

# Parse inputs and make destination folder
folder=sys.argv[1]
if folder[-1]!="/":folder+="/"

if not os.path.isdir(folder+"movie/"): os.mkdir(folder+"movie/")

stim = 0
pre = 1.0
post = 1.0
shift_g = [0,0]
latency_shift = 0
show_stim = "--no-show-stim" not in sys.argv
frame_rate = 4
for s in sys.argv[1:]:
    sa = s.split(":")
    if sa[0]=="--stim":stim=int(sa[1])
    if sa[0]=="--pre":pre=float(sa[1])
    if sa[0]=="--post":post=float(sa[1])
    if sa[0]=="--shift-g":shift_g=[int(a) for a in sa[1].split(",")]
    if sa[0]=="--frame-rate":frame_rate=int(sa[1])
    
# Load objects
rec = wormdm.data.recording(folder)
events = rec.get_events()['optogenetics']

info = wormb.match.load_match_parameters(folder)
ref_index = info['ref_index']

cerv = wormb.Brains.from_file(folder)
labels = cerv.get_labels(ref_index,attr=False)

fconn = pp.Fconn.from_file(folder)
v0 = events['index'][stim] - int(pre/rec.Dt)
v1 = events['index'][stim] + int(post/rec.Dt)

rec.load(startVolume=v0,nVolume=v1-v0)
volFrame0 = rec.volumeFirstFrame[v0:v1+1].copy()-rec.volumeFirstFrame[v0]
ZZ = rec.ZZ[v0:v1]

# Make first round of images and projections to crop and initialize shapes
z_proj_g = np.max(rec.frame[volFrame0[0]:volFrame0[1],1],axis=0)
z_proj_r = np.max(rec.frame[volFrame0[0]:volFrame0[1],0],axis=0)

_, r_c_r = mf.geometry.draw.crop_image(z_proj_r,folder,fname="rectangle_movie.txt",x_axis=1,y_axis=0,return_all=True)
# Transform rectangle to green reference frame and save it to file
r_c_g_0 = wormdm.data.redToGreen(np.array([[-1,r_c_r[0,1],r_c_r[0,0]]]), folder=folder)[0]
r_c_g_1 = wormdm.data.redToGreen(np.array([[-1,r_c_r[1,1],r_c_r[1,0]]]), folder=folder)[0]
r_c_g = [[r_c_g_0[1],r_c_g_0[2]],[r_c_g_1[1],r_c_g_1[2]]]
np.savetxt(folder+"rectangle_movie_green.txt",r_c_g,fmt="%d")
_, r_c_g = mf.geometry.draw.crop_image(z_proj_g,folder,fname="rectangle_movie_green.txt",x_axis=1,y_axis=0,return_all=True)

clp_frame_r = np.clip(rec.frame[volFrame0[0]:volFrame0[1],0].astype(float),100,None)
clp_frame_g = np.clip(rec.frame[volFrame0[0]:volFrame0[1],1].astype(float),100,None)
clp_frame_r = mf.geometry.draw.crop_image(clp_frame_r,folder,fname="rectangle_movie.txt",x_axis=2,y_axis=1)
clp_frame_g = mf.geometry.draw.crop_image(clp_frame_g,folder,fname="rectangle_movie_green.txt",x_axis=2,y_axis=1)
x_proj_r = np.max(clp_frame_r,axis=-1) #z is the 0th index
x_proj_g = np.max(clp_frame_g,axis=-1)

x_proj_r_interp = np.zeros((x_proj_r.shape[0]*3,x_proj_r.shape[1]))
x_proj_g_interp = np.zeros((x_proj_r.shape[0]*3,x_proj_r.shape[1]))
zinterp = np.arange(x_proj_r_interp.shape[0])/x_proj_r_interp.shape[0]*(max(ZZ[0])-min(ZZ[0]))+min(ZZ[0])

zvol = None
voldir = None

# Calculate the aspect ratio such that you can have the same scale on x and z
Dz = np.max(zinterp)-np.min(zinterp)
Dx = x_proj_r_interp.shape[1]
ar = Dz/Dx

fig = plt.figure(1,figsize=(15,15))

# Filenames of temporary images
tmp_fnames = []

for v in np.arange(v1-v0):
    fig.clear()
    gs = fig.add_gridspec(20, 10)
    # Use the aspect ratio to determine the relative size of the xy and yz projections
    ax_zpan = int(10*(1-ar))
    ax1 = fig.add_subplot(gs[0:10,:ax_zpan])
    ax2 = fig.add_subplot(gs[0:10,ax_zpan:],sharey=ax1)
    ax3 = fig.add_subplot(gs[10:20,:ax_zpan])
    ax4 = fig.add_subplot(gs[10:20,ax_zpan:],sharey=ax3)
    
    # Clip out noise
    clp_frame_r = np.clip(rec.frame[volFrame0[v]:volFrame0[v+1],0].astype(float),110,None)
    clp_frame_g = np.clip(rec.frame[volFrame0[v]:volFrame0[v+1],1].astype(float),110,None)
    
    # Crop image
    clp_frame_r = mf.geometry.draw.crop_image(clp_frame_r,folder,fname="rectangle_movie.txt",x_axis=2,y_axis=1)
    clp_frame_g = mf.geometry.draw.crop_image(clp_frame_g,folder,fname="rectangle_movie_green.txt",x_axis=2,y_axis=1)
    
    # Max z projection
    z_proj_r = np.max(clp_frame_r[4:-4],axis=0)
    z_proj_g = np.max(clp_frame_g[4:-4],axis=0)
    
    # X projection (max or sum)
    max_projection = True
    single_slice = False
    x_proj_r_old = np.copy(x_proj_r)
    x_proj_g_old = np.copy(x_proj_g)
    if max_projection:
        x_proj_r = np.max(clp_frame_r,axis=-1)
        x_proj_g = np.max(clp_frame_g,axis=-1)
    elif single_slice:
        x_proj_r = clp_frame_r[...,270]
        x_proj_g = clp_frame_g[...,270]
    
    # Initialize zvol_old and voldir_old for averaging 2 volumes
    if zvol is not None:
        zvol_old = np.copy(zvol)
        voldir_old = np.copy(voldir)
    else:
        zvol_old = None
        voldir_old = None
    
    # Calculate volume direction and correct for half the step (upwards or 
    # downwards exposure).
    voldir = int(np.sign(ZZ[v][10]-ZZ[v][0]))
    zvol = np.array(ZZ[v])
    zvol[1:] += -np.diff(zvol)/2.
    
    for y in np.arange(x_proj_r_interp.shape[1]):
        # If it's not the first volume, compute a moving average of the yz projections
        if zvol_old is not None:
            x_proj_r_interp[:,y] = 0.5*np.interp(zinterp, zvol[::voldir], x_proj_r[:,y][::voldir])
            x_proj_r_interp[:,y] += 0.5*np.interp(zinterp, zvol_old[::voldir_old], x_proj_r_old[:,y][::voldir_old])
            x_proj_g_interp[:,y] = 0.5*np.interp(zinterp, np.array(zvol)[::voldir], x_proj_g[:,y][::voldir])
            x_proj_g_interp[:,y] += 0.5*np.interp(zinterp, zvol_old[::voldir_old], x_proj_g_old[:,y][::voldir_old])
        else:
            x_proj_r_interp[:,y] = np.interp(zinterp, zvol[::voldir], x_proj_r[:,y][::voldir])
            x_proj_g_interp[:,y] = np.interp(zinterp, np.array(zvol)[::voldir], x_proj_g[:,y][::voldir])
        
    # Store vmax from the first volume
    if v == 0:
        vmax_z_r = np.max(z_proj_r)
        vmax_z_g = np.max(z_proj_g)#*0.4
        vmax_x_r = np.max(x_proj_r_interp)
        vmax_x_g = np.max(x_proj_g_interp)#*0.4
    
    # Plots    
    ax1.imshow(z_proj_r,aspect="auto",cmap="magma",interpolation="none",vmax=vmax_z_r)
    ax2.imshow(x_proj_r_interp.T,cmap="magma",vmax=vmax_x_r,interpolation="none",aspect="auto") #vmin=vmin_x_r, aspect="auto"
    ax3.imshow(z_proj_g,aspect="auto",cmap="jet",interpolation="none",vmax=vmax_z_g)
    ax4.imshow(x_proj_g_interp.T,cmap="jet",vmin=110,vmax=vmax_x_g,interpolation="none",aspect="auto") #vmin=vmin_x_g,,aspect="auto"
    
    # Add the crosshairs
    lbl = ""
    for ie in np.arange(len(events['index'])):
        ve = events['index'][ie]
        x,y,z = events['properties']['target'][ie]
        z_g,y_g,x_g = wormdm.data.redToGreen(np.array([[z,y,x]]), folder=folder)[0]
        y -= r_c_r[0,0] 
        x -= r_c_r[0,1]
        y_g -= r_c_g[0,0] + shift_g[0]
        x_g -= r_c_g[0,1] + shift_g[1]
        iz_crosshair = np.argmin(np.abs(z-zinterp))
            
        if 0<=v+v0-ve<15*4 and show_stim and (x>0 and y>0 and x_g>0 and y_g>0):
            ax1.axvline(x,c='g')
            ax1.axhline(y,c='g')
            ax2.axvline(iz_crosshair,c='g',lw=0.8)
            ax2.axhline(y,c='g',lw=0.8)
            ax3.axvline(x_g,c='r')
            ax3.axhline(y_g,c='r')
            ax4.axvline(iz_crosshair,c='r',lw=0.8)
            ax4.axhline(y_g,c='r',lw=0.8)
            
            if fconn.stim_neurons[ie]>=0:
                lbl = labels[fconn.stim_neurons[ie]]
            elif fconn.stim_neurons[ie]==-3:
                lbl = fconn.stim_neurons_compl_labels[ie]
    
    ax1.set_title("t: "+str(np.around((v+v0)*rec.Dt,2))+" s   "+lbl)
            
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    plt.savefig(folder+"movie/%04d.png" % v,dpi=200,bbox_inches="tight")#,vmin=100,vmax=200)
    tmp_fnames.append(folder+"movie/%04d.png" % v)

os.chdir(folder+"movie/") #
#subprocess.call([
#    'ffmpeg', '-framerate', '4', '-i', '%04d.png', '-r', '30', '-pix_fmt', 'yuv420p', '-vf',"scale='bitand(oh*dar,65534)':'min(720,ih)'", '-vb','20M','-c:v','libx264','-y', #rgb
#    'stim_'+str(stim)+'.mp4'
#])
subprocess.call([
    'ffmpeg', '-framerate', str(frame_rate), '-i', '%04d.png', '-r', '30', '-pix_fmt','yuv420p','-vf',"scale='bitand(oh*dar,65534)':'min(720,ih)'",'-profile:v','baseline','-level','3.0','-c:v','libx264','-y','./stim_'+str(stim)+'.mp4'
])
#for fname in tmp_fnames:
#    os.remove(fname)
