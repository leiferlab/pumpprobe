import matplotlib.pyplot as plt, numpy as np


def make_alphacolorbar(cax,vmin,vmax,tickstep,
                       alphamin,alphamax,nalphaticks,
                       cmap="viridis",bg_gray=0.2,around=0,lbl_lg=False):
    '''Make a 2D colorbar with transparency on one axis.
    
    Parameters
    ----------
    cax: matplotlib axis
        Axis in which to draw the colorbar.
    vmin, vmax: float
        Minimum and maximum values of the colormap axis (y).
    tickstep: float
        Step of the ticks on the colormap axis (y).
    alphamin, alphamax: float
        Minimum and maximum alphas.
    nalphaticks: int
        Number of ticks on the alpha axis (x).
    cmap: matplotlib-accepted colormap (optional)
        Colormap.
    around: int (optional)
        np.around -ing precision for ticks. Default: 0.
    '''
    
    nticks = int((vmax-vmin)/tickstep)
    
    alphacm1 = np.zeros((200,200))
    alphacm1[:] = np.arange(alphacm1.shape[0])[:,None]
    alphacm2 = np.zeros_like(alphacm1)
    alphacm2[:] = (np.arange(alphacm1.shape[1])/alphacm1.shape[1])[None,:]
    
    background = np.ones_like(alphacm1)*bg_gray
    
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    cax.imshow(background,cmap="Greys",vmin=0,vmax=1)
    cax.imshow(alphacm1,alpha=alphacm2,aspect="auto",cmap=cmap,
               interpolation="nearest",origin="lower")

    cax.set_xticks([0,alphacm1.shape[1]])
    cax.set_xticklabels([str(alphamin),str(alphamax)])
    yticks = np.linspace(0,alphacm1.shape[0],nticks+1)
    cax.set_yticks(yticks)
    d = (vmax-vmin)/nticks
    yticklbl = [str(np.around(i*d+vmin,around)) for i in np.arange(len(yticks))]
    if lbl_lg: 
        yticklbl[0] = "<"+yticklbl[0]
        yticklbl[-1] = ">"+yticklbl[-1]
    cax.set_yticklabels(yticklbl)
    cax.set_xlim(0,alphacm1.shape[1])
    cax.set_ylim(0,alphacm1.shape[0])
    
def make_colorbar(cax,vmin,vmax,tickstep,cmap="viridis",label="q",totsize=200,**kwargs):
    nticks = int((vmax-vmin)/tickstep)
    x = np.array([np.linspace(vmin,vmax,totsize)])
    
    cax.imshow(x.T,aspect="auto",**kwargs)
    
    cax.set_xticks([])
    yticks = np.linspace(0,x.shape[1],nticks+1)
    cax.set_yticks(yticks)
    
    d = (vmax-vmin)/nticks
    ylbl = [str(np.around(i*d+vmin,2)) for i in np.arange(len(yticks))]
    cax.set_yticklabels(ylbl)
    
    cax.set_ylabel(label)
