import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp
from scipy.optimize import lsq_linear

plt.rc('xtick',labelsize=5)
plt.rc('ytick',labelsize=5)

funa = pp.Funatlas(merge_bilateral=False,merge_dorsoventral=False,merge_numbered=False,)
                                     
ids = funa.head_ids

funa.load_aconnectome_from_file(chem_th=0,gap_th=0)
conn = funa.reduce_to_head(funa.aconn_chem+funa.aconn_gap)
medabsconn = np.median(np.abs(conn[conn!=0]))
print(medabsconn)

#excl_ds = np.arange(42)
excl_ds = [None]
n_excl_ds = len(excl_ds)
n_head_neurons = len(funa.head_ids)
f_ = np.zeros((n_excl_ds,n_head_neurons,n_head_neurons))
F_ = np.zeros((n_excl_ds,n_head_neurons,n_head_neurons))
for i_excl_ds in np.arange(n_excl_ds):
    if excl_ds[0] is not None:
        txt_fname = "funatlas_intensity_map_excl_"+str(i_excl_ds)+".txt"
    else:
        txt_fname = "funatlas_intensity_map_cache.txt"
        txt_fname_q = "funatlas_intensity_map_cache_q.txt"
    F = np.loadtxt("/projects/LEIFER/francesco/funatlas/"+txt_fname)
    F[np.isnan(F)] = 0.0
    q = np.loadtxt("/projects/LEIFER/francesco/funatlas/"+txt_fname_q)
    F[q>0.05] = 0.0
    F = funa.reduce_to_head(F)
    
    # Normalizing 
    medabsF = np.median(np.abs(F[F!=0]))
    print(medabsF)
    F = F/medabsF*medabsconn
    
    F_[i_excl_ds] = F
    f = np.zeros_like(F)

    #Build the variables of the linear system
    A = np.copy(F)
    A[np.diag_indices(F.shape[0])] = 1.0
    #Build the variable bounds
    bounds_up = np.empty(F.shape[0])
    bounds_lo = np.empty(F.shape[0])

    for i in np.arange(F.shape[0]):
        b = F[i]
        bounds_up[:] = np.inf
        bounds_lo[:] = -np.inf
        bounds_up[conn[i]==0] = 1e-6
        bounds_lo[conn[i]==0] = -1e-6
        bounds = (bounds_lo,bounds_up)
        
        res = lsq_linear(A,b,bounds)
        f[i] = res.x
        
        f_[i_excl_ds,i] = f[i]
    
    if excl_ds[0] is None:
        np.savetxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_inverted.txt",f[0])

    r_conn_F = np.corrcoef([np.ravel(conn),np.ravel(F)])[0,1]
    r_conn_f = np.corrcoef([np.ravel(np.abs(conn)),np.ravel(np.abs(f))])[0,1]
    r_f_F = np.corrcoef([np.ravel(f),np.ravel(F)])[0,1]
    
    # Fraction of f that turned out to be zero despite not being constrained
    # to be zero
    silenced = []
    th =[]
    for logth in np.arange(-16,-1,1):
        th_ = 10.0**logth
        sil = np.sum(np.abs(f[conn!=0.0])<=th_)/np.sum(conn!=0.0)
        th.append(th_)
        silenced.append(sil)
        
    # Relative variation of the absolute weight of connections between the
    # anatomical connectome and the inverted f
    af = np.abs(f)
    maf = np.max(af)
    mconn = np.max(conn)
    dw = np.abs(conn/mconn-af/maf)
    dww = np.zeros_like(dw)
    dww[conn!=0] = dw[conn!=0]/(conn[conn!=0]/mconn)
    dww[conn==0] = np.nan
    
    print(r_conn_F,r_conn_f,r_f_F)

    if excl_ds[0] is None:
        fig1 = plt.figure(1,figsize=(12,10))
        ax1 = fig1.add_subplot(111)
        ax1.imshow(F)
        ax1.set_xticks(np.arange(len(ids)))
        ax1.set_yticks(np.arange(len(ids)))
        ax1.set_xticklabels(ids,rotation=90)
        ax1.set_yticklabels(ids)
        ax1.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
        ax1.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
        fig1.tight_layout()

        fig2 = plt.figure(2,figsize=(12,10))
        ax2 = fig2.add_subplot(111)
        ax2.imshow(f)
        ax2.set_xticks(np.arange(len(ids)))
        ax2.set_yticks(np.arange(len(ids)))
        ax2.set_xticklabels(ids,rotation=90)
        ax2.set_yticklabels(ids)
        ax2.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
        ax2.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
        fig2.tight_layout()
        
        plt.rc('xtick',labelsize=10)
        plt.rc('ytick',labelsize=10)
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111)
        ax3.plot(th,silenced)
        ax3.set_xscale('log')
        ax3.set_xlabel("threshold")
        ax3.set_ylabel("fraction of zero-weight connections\nthat could be nonzero")
        fig3.tight_layout()
        
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        ax4.hist(dww[~np.isnan(dww)],bins=100)
        ax4.set_xlabel("relative change in weight")
        ax4.set_ylabel("number of connections")
        
        plt.show()

if excl_ds[0] is None:
    # Reexpand to entire body
    f_body = np.zeros((funa.n_neurons,funa.n_neurons))
    for i in np.arange(len(funa.head_ai)):
        for j in np.arange(len(funa.head_ai)):
            f_body[funa.head_ai[i],funa.head_ai[j]] = f[i,j]
    np.savetxt("/projects/LEIFER/francesco/funatlas/funatlas_intensity_map_inverted.txt",f_body)
    plt.figure(6)
    plt.imshow(f_body)
    plt.show()

if excl_ds[0] is not None:
    r_fi_fk = np.zeros((n_excl_ds,n_excl_ds))
    r_Fi_Fk = np.zeros((n_excl_ds,n_excl_ds))
    for i_excl_ds in np.arange(n_excl_ds):
        for k_excl_ds in np.arange(n_excl_ds):
            r_fi_fk[i_excl_ds,k_excl_ds] = np.corrcoef([np.ravel(f_[i_excl_ds]),np.ravel(f_[k_excl_ds])])[0,1]
            r_Fi_Fk[i_excl_ds,k_excl_ds] = np.corrcoef([np.ravel(F_[i_excl_ds]),np.ravel(F_[k_excl_ds])])[0,1]

    print(np.ravel(r_fi_fk))
    print(np.any(np.isnan(r_fi_fk)))
    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(111)
    ax3.plot(np.ravel(r_Fi_Fk),np.ravel(r_fi_fk),'o')
    ax3.set_xlabel("corr of F[i] and F[k]")
    ax3.set_ylabel("corr of f[i] and f[k]")
    fig3.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/F_to_f_inversion_steady_state.png",bbox_inches="tight")
    plt.show()
