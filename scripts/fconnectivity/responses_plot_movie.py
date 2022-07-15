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

folder=sys.argv[1]
if folder[-1]!="/":folder+="/"

if not os.path.isdir(folder+"movie/"): os.mkdir(folder+"movie/")

stim = 0
pre = 1.0
post = 1.0
shift_g = [0,0]
latency_shift = 0
show_stim = "--no-show-stim" not in sys.argv
for s in sys.argv[1:]:
    sa = s.split(":")
    if sa[0]=="--stim":stim=int(sa[1])
    if sa[0]=="--pre":pre=float(sa[1])
    if sa[0]=="--post":post=float(sa[1])
    if sa[0]=="--shift-g":shift_g=[int(a) for a in sa[1].split(",")]
    
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

z_proj_g = np.max(rec.frame[volFrame0[0]:volFrame0[1],1],axis=0)
z_proj_r = np.max(rec.frame[volFrame0[0]:volFrame0[1],0],axis=0)

print(folder+"rectangle_movie.txt")
_, r_c_r = mf.geometry.draw.crop_image(z_proj_r,folder,fname="rectangle_movie.txt",x_axis=1,y_axis=0,return_all=True)
_, r_c_g = mf.geometry.draw.crop_image(z_proj_g,folder,fname="rectangle_movie_green.txt",x_axis=1,y_axis=0,return_all=True)

clp_frame_r = np.clip(rec.frame[volFrame0[0]:volFrame0[1],0].astype(float),100,None)
clp_frame_g = np.clip(rec.frame[volFrame0[0]:volFrame0[1],1].astype(float),100,None)
clp_frame_r = mf.geometry.draw.crop_image(clp_frame_r,folder,fname="rectangle_movie.txt",x_axis=2,y_axis=1)
clp_frame_g = mf.geometry.draw.crop_image(clp_frame_g,folder,fname="rectangle_movie_green.txt",x_axis=2,y_axis=1)
x_proj_r = np.max(clp_frame_r,axis=-1)
x_proj_g = np.max(clp_frame_g,axis=-1)
x_proj_r_0 = np.copy(x_proj_r)#[:,r_c_r[0,0]:r_c_r[1,0]])
x_proj_g_0 = np.copy(x_proj_g)#[:,r_c_g[0,0]:r_c_g[1,0]])

fig = plt.figure(1,figsize=(15,15))

z_corrections_avg = []
avg_z_step = 0.15*rec.zUmOverV*rec.framePixelPerUm#np.median(np.abs(np.diff(ZZ[v_ref])))
for v_ref in np.arange(v1-v0):
    z_corrections = []
    for v in np.arange(v1-v0):
        voldir = np.sign(ZZ[v][10]-ZZ[v][0])
        updownz = -voldir*avg_z_step/2
        for iz in np.arange(len(ZZ[v])):
            iz_corr = np.argmin(np.abs(ZZ[v][iz]+updownz-np.array(ZZ[v_ref])))
            z_corrections.append(ZZ[v][iz]-ZZ[v_ref][iz_corr])
    z_corrections_avg.append(np.average(np.abs(z_corrections)))
v_ref = np.argmin(z_corrections_avg)
print("v_ref",v_ref)
print("avg_z_step",avg_z_step)

tmp_fnames = []

for v in np.arange(v1-v0):
    fig.clear()
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0:1,0:2])
    ax2 = fig.add_subplot(gs[0:1,2:3],sharey=ax1)
    ax3 = fig.add_subplot(gs[1:2,0:2])
    ax4 = fig.add_subplot(gs[1:2,2:3],sharey=ax3)
    
    clp_frame_r = np.clip(rec.frame[volFrame0[v]:volFrame0[v+1],0].astype(float),110,None)
    clp_frame_g = np.clip(rec.frame[volFrame0[v]:volFrame0[v+1],1].astype(float),110,None)
    
    clp_frame_r = mf.geometry.draw.crop_image(clp_frame_r,folder,fname="rectangle_movie.txt",x_axis=2,y_axis=1)
    clp_frame_g = mf.geometry.draw.crop_image(clp_frame_g,folder,fname="rectangle_movie_green.txt",x_axis=2,y_axis=1)
    
    z_proj_r = np.max(clp_frame_r[4:-4],axis=0)
    z_proj_g = np.max(clp_frame_g[4:-4],axis=0)
    
    max_projection = True
    single_slice = False
    if max_projection:
        x_proj_r = np.max(clp_frame_r,axis=-1)
        x_proj_g = np.max(clp_frame_g,axis=-1)
    elif single_slice:
        x_proj_r = clp_frame_r[...,270]#r_c_r[0,0]+90]#137
        x_proj_g = clp_frame_g[...,270]#r_c_r[0,0]+90]
    
    #z_proj_r = mf.geometry.draw.crop_image(z_proj_r,folder,fname="rectangle_movie.txt",x_axis=1,y_axis=0)
    #x_proj_r = x_proj_r[:,r_c_r[0,0]:r_c_r[1,0]]
    #z_proj_g = mf.geometry.draw.crop_image(z_proj_g,folder,fname="rectangle_movie_green.txt",x_axis=1,y_axis=0)
    #x_proj_g = x_proj_g[:,r_c_g[0,0]:r_c_g[1,0]]
    
    
    x_proj_r_new = np.ones((x_proj_r_0.shape[0]+5,x_proj_r_0.shape[1]))*0 #was +5
    x_proj_g_new = np.ones((x_proj_g_0.shape[0]+5,x_proj_g_0.shape[1]))*0  
    
    zdist_from_matched = []
    voldir = np.sign(ZZ[v][10]-ZZ[v][0])
    updownz = -voldir*avg_z_step/2
    for izref in np.arange(len(ZZ[v_ref])):
        z_ref = ZZ[v_ref][izref]
        iz_corrs = np.argsort(np.abs(np.array(ZZ[v])+updownz-z_ref))
        z0 = ZZ[v][iz_corrs[0]]
        zdist_from_matched.append(z0-z_ref)
        x_proj_r_new[izref] = x_proj_r[iz_corrs[0]]
        x_proj_g_new[izref] = x_proj_g[iz_corrs[0]]
        '''
        if np.sign(z_ref-z1)==np.sign(z_ref-z0):
            # You're at an extremum of the scan
            x_proj_r_new[izref] = x_proj_r[iz_corrs[0]]
            x_proj_g_new[izref] = x_proj_g[iz_corrs[0]]
        else:
            w0 = (1-abs(z_ref-z0)/abs(z0-z1))
            w1 = (1-abs(z_ref-z1)/abs(z0-z1))
            
            x_proj_r_new[izref] = w0*x_proj_r[iz_corrs[0]]+\
                                  w1*x_proj_r[iz_corrs[1]]
            x_proj_g_new[izref] = w0*x_proj_g[iz_corrs[0]]+\
                                  w1*x_proj_g[iz_corrs[1]]'''
        
    # Store vmax from the first volume
    if v == 0:
        vmax_z_r = np.max(z_proj_r)
        vmax_z_g = np.max(z_proj_g)#*0.4
        vmax_x_r = np.max(x_proj_r_new)
        vmax_x_g = np.max(x_proj_g_new)#*0.4
        #vmin_x_r = np.median(x_proj_r_new)
        #vmin_x_g = np.median(x_proj_g_new)
    
    ax1.imshow(z_proj_r,aspect="auto",cmap="magma",interpolation="none",vmax=vmax_z_r)
    # [:,15:20] for /projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20220120/pumpprobe_20220120_104715/movie with single neuron
    # [:,12:20]  /projects/LEIFER/francesco/pumpprobe/instrument_characterization/zscan_adjustment/20220127/pumpprobe_20220127_164629/
    ax2.imshow(x_proj_r_new.T,cmap="magma",vmax=vmax_x_r,interpolation="none",aspect="auto") #vmin=vmin_x_r, aspect="auto"
    ax3.imshow(z_proj_g,aspect="auto",cmap="jet",interpolation="none",vmax=vmax_z_g)
    ax4.imshow(x_proj_g_new.T,cmap="jet",vmin=110,vmax=vmax_x_g,interpolation="none",aspect="auto") #vmin=vmin_x_g,,aspect="auto"
    
    lbl = ""
    for ie in np.arange(len(events['index'])):
        ve = events['index'][ie]
        x,y,z = events['properties']['target'][ie]
        z_g,y_g,x_g = wormdm.data.redToGreen(np.array([[z,y,x]]), folder=folder)[0]
        y -= r_c_r[0,0] 
        x -= r_c_r[0,1]
        y_g -= r_c_g[0,0] + shift_g[0]
        x_g -= r_c_g[0,1] + shift_g[1]
        iz_corr = np.argmin(np.abs(z-np.array(ZZ[v_ref])))#+delta_z
        #if 0<v+v0-ve<10:
        #    ax1.plot(x,y,'og',markersize=7)
        #    ax2.plot(iz_corr,y,'og',markersize=7)
        #    ax3.plot(x_g,y_g,'or',markersize=7)
        #    ax4.plot(iz_corr,y_g,'or',markersize=7)
            
        if 0<v+v0-ve<15*4 and show_stim and (x>0 and y>0 and x_g>0 and y_g>0):
            ax1.axvline(x,c='g')
            ax1.axhline(y,c='g')
            ax2.axvline(iz_corr,c='g',lw=0.8)
            ax2.axhline(y,c='g',lw=0.8)
            ax3.axvline(x_g,c='r')
            ax3.axhline(y_g,c='r')
            ax4.axvline(iz_corr,c='r',lw=0.8)
            ax4.axhline(y_g,c='r',lw=0.8)
            
            if fconn.stim_neurons[ie]>=0:
                lbl = labels[fconn.stim_neurons[ie]]
            elif fconn.stim_neurons[ie]==-3:
                lbl = fconn.stim_neurons_compl_labels[ie]
    
    ax1.set_title("t: "+str(np.around((v+v0)*rec.Dt,2))+" s   "+lbl)#+"   zdist_from_matched max "+str(np.around(np.max(np.abs(zdist_from_matched)),4)))
            
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    #ax2.set_xlim(2,23)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    #ax4.set_xlim(2,23)
    
    '''#To show a single neuron for /projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211104/pumpprobe_20211104_163944/ --stim:21 --pre:10 --post:30 --shift-g:4,10
    ax2.set_xlim(4,12)
    ax2.set_ylim(171,160)
    ax4.set_xlim(4,12)
    ax4.set_ylim(171,160)'''
    
    plt.savefig(folder+"movie/%04d.png" % v,dpi=200,bbox_inches="tight")#,vmin=100,vmax=200)
    #plt.show()
    tmp_fnames.append(folder+"movie/%04d.png" % v)

os.chdir(folder+"movie/") #
#subprocess.call([
#    'ffmpeg', '-framerate', '4', '-i', '%04d.png', '-r', '30', '-pix_fmt', 'yuv420p', '-vf',"scale='bitand(oh*dar,65534)':'min(720,ih)'", '-vb','20M','-c:v','libx264','-y', #rgb
#    'stim_'+str(stim)+'.mp4'
#])
subprocess.call([
    'ffmpeg', '-framerate', '4', '-i', '%04d.png', '-r', '30', '-pix_fmt','yuv420p','-vf',"scale='bitand(oh*dar,65534)':'min(720,ih)'",'-profile:v','baseline','-level','3.0','-c:v','libx264','-y','./stim_'+str(stim)+'.mp4'
])
#for fname in tmp_fnames:
#    os.remove(fname)
