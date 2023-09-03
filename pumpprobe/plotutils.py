import numpy as np
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import matplotlib.ticker

c_in_wt = "C0"
c_not_in_wt = "C1"
c_wt = "C2"
c_unc31 = "C9"

def make_alphacolorbar(cax,vmin,vmax,tickstep,
                       alphamin,alphamax,nalphaticks,
                       cmap="viridis",bg_gray=0.2,around=0,lbl_lg=False,
                       lbl_g=False,alphaticklabels=None):
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
    if alphaticklabels is None:
        cax.set_xticklabels([str(alphamin),str(alphamax)])
    else:
        cax.set_xticklabels([alphaticklabels[0],alphaticklabels[1]])
    yticks = np.linspace(0,alphacm1.shape[0],nticks+1)
    cax.set_yticks(yticks)
    d = (vmax-vmin)/nticks
    yticklbl = [str(np.around(i*d+vmin,around)) for i in np.arange(len(yticks))]
    if lbl_lg: 
        yticklbl[0] = "<"+yticklbl[0]
        yticklbl[-1] = ">"+yticklbl[-1]
    if lbl_g:
        yticklbl[-1] = ">"+yticklbl[-1]
    cax.set_yticklabels(yticklbl)
    cax.set_xlim(0,alphacm1.shape[1])
    cax.set_ylim(0,alphacm1.shape[0])
    
def make_colorbar(cax,vmin,vmax,tickstep,cmap="viridis",label="q",totsize=200,**kwargs):
    nticks = int((vmax-vmin)/tickstep)
    x = np.array([np.linspace(vmin,vmax,totsize)])
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")
    
    cax.imshow(x.T,aspect="auto",cmap=cmap,**kwargs)
    
    cax.set_xticks([])
    yticks = np.linspace(0,x.shape[1],nticks+1)
    cax.set_yticks(yticks)
    
    d = (vmax-vmin)/nticks
    ylbl = [str(np.around(i*d+vmin,2)) for i in np.arange(len(yticks))]
    cax.set_yticklabels(ylbl)
    
    cax.set_ylabel(label)
    
def plot_linlog(x,y,linlog_edge,fig,axis="y",color="C0",size=0.5,xlim=None):
    
    if axis=="y":
        spec = gridspec.GridSpec(ncols=1, nrows=4, figure=fig)
        ax1 = fig.add_subplot(spec[0,0])
        ax2 = fig.add_subplot(spec[1:,0])
        fig.subplots_adjust(hspace=0.05)
    elif axis=="x":
        spec = gridspec.GridSpec(ncols=4, nrows=1, figure=fig)
        ax1 = fig.add_subplot(spec[0,1:])
        ax2 = fig.add_subplot(spec[0,0])
        fig.subplots_adjust(wspace=0.02)
        
    ax1.scatter(x, y, color=color,s=size)
    ax2.scatter(x, y, color=color,s=size)
    
    ax1.minorticks_on()
    ax2.minorticks_on()
    
    if axis=="y":
        ax2.set_ylim(None,linlog_edge)
        ax1.set_ylim(linlog_edge,None)
        ax1.set_yscale("log")
        
        ax2.xaxis.tick_bottom()
   
        ax1.spines.bottom.set_visible(False)
        ax1.spines.top.set_visible(False)
        ax1.spines.right.set_visible(False)
        ax1.tick_params(labeltop=False,bottom=False,top=False)
        
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
        ax2.plot([0], [1], transform=ax2.transAxes, **kwargs)
        
    elif axis=="x":
        ax2.set_xlim(xlim,linlog_edge)
        ax1.set_xlim(linlog_edge,None)
        ax1.set_xscale("log")
        
        ax2.yaxis.tick_left()
        ax2.spines.right.set_visible(False)
        ax2.spines.top.set_visible(False)
        ax2.tick_params(labelright=False)
        ax2.tick_params(axis='x', which='minor', bottom=True)
        ax2.tick_params(axis='y', which='minor', left=False)
   
        ax1.spines.left.set_visible(False)
        ax1.spines.right.set_visible(False)
        ax1.spines.top.set_visible(False)
        ax1.tick_params(labelright=False,labelleft=False,left=False,right=False,bottom=True)
        ax1.tick_params(axis='x', which='minor', bottom=True)
        ax1.tick_params(axis='y', which='minor', left=False)
        locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8),numticks=12)
        ax1.xaxis.set_minor_locator(locmin)
        ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        
        d = 2.  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([0], [0], transform=ax1.transAxes, **kwargs)
        ax2.plot([1], [0], transform=ax2.transAxes, **kwargs)
    
    

    return ax1, ax2
    
def scatter_hist(x, y, ax, ax_histx, ax_histy, label="", binwidth=0.25, binsx=None, binsy=None, alpha_scatter=1,alpha_hist=1,color=None,hist_density=False,**hist_kwargs):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, label=label, alpha=alpha_scatter, color=color)

    # now determine nice limits by hand:
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    if binsx is None: binsx=bins
    if binsy is None: binsy=bins
    
    ax_histx.hist(x, bins=binsx,alpha=alpha_hist,color=color,**hist_kwargs)
    ax_histy.hist(y, bins=binsy,orientation='horizontal',alpha=alpha_hist,color=color,**hist_kwargs)
    
def simple_beeswarm(y, nbins=None):
    """
    Returns x coordinates for the points in ``y``, so that plotting ``x`` and
    ``y`` results in a bee swarm plot.
    https://stackoverflow.com/questions/36153410/how-to-create-a-swarm-plot-with-matplotlib
    """
    y = np.asarray(y)
    if nbins is None:
        nbins = len(y) // 6

    # Get upper bounds of bins
    x = np.zeros(len(y))
    ylo = np.min(y)
    yhi = np.max(y)
    dy = (yhi - ylo) / nbins
    ybins = np.linspace(ylo + dy, yhi - dy, nbins - 1)

    # Divide indices into bins
    i = np.arange(len(y))
    ibs = [0] * nbins
    ybs = [0] * nbins
    nmax = 0
    for j, ybin in enumerate(ybins):
        f = y <= ybin
        ibs[j], ybs[j] = i[f], y[f]
        nmax = max(nmax, len(ibs[j]))
        f = ~f
        i, y = i[f], y[f]
    ibs[-1], ybs[-1] = i, y
    nmax = max(nmax, len(ibs[-1]))

    # Assign x indices
    dx = 1 / (nmax // 2)
    for i, y in zip(ibs, ybs):
        if len(i) > 1:
            j = len(i) % 2
            i = i[np.argsort(y)]
            a = i[j::2]
            b = i[j+1::2]
            x[a] = (0.5 + j / 3 + np.arange(len(b))) * dx
            x[b] = (0.5 + j / 3 + np.arange(len(b))) * -dx

    return x
