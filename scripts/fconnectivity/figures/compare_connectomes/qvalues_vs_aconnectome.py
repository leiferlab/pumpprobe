import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp
from scipy.optimize import curve_fit

plt.rc("xtick",labelsize=18)
plt.rc("ytick",labelsize=18)
plt.rc("axes",labelsize=18)

to_paper = "--to-paper" in sys.argv
cumulative = "--cumulative" in sys.argv
exclude_white = "--exclude-white" in sys.argv
use_pvalues = "--use-pvalues" in sys.argv
use_qvalues = not use_pvalues
#req_auto_response = "--req-auto-response" in sys.argv

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="mutant",
                enforce_stim_crosscheck=False,verbose=False)
                
# OLD QVALUES
#occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=req_auto_response)
#occ3 = funa.get_observation_matrix(req_auto_response=req_auto_response)
#qvalues = funa.get_qvalues(occ1,occ3,False)
# NEW QVALUES
_, inclall_occ2 = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
if use_qvalues:
    qvalues = funa.get_kolmogorov_smirnov_q(inclall_occ2)
    y_label = "q value of connection"
elif use_pvalues:
    qvalues,_,_ = funa.get_kolmogorov_smirnov_p(inclall_occ2)
    y_label = "p value of connection"


act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_bilateral_merged.txt")
print("using sign2")

aconn_chem, aconn_gap = funa.get_aconnectome_from_file(
                            gap_th=0,chem_th=0,exclude_white=exclude_white,
                            average=(not cumulative))
                                            
aconn = aconn_chem+aconn_gap

qval_y = qvalues[~np.isnan(qvalues)]
maxqval = np.max(qval_y)
aconn_x = aconn[~np.isnan(qvalues)]
act_conn_x = act_conn[~np.isnan(qvalues)]

def fitf(x,a,b=(1-maxqval)):
    return a*x+b
    
def fitf_jac(x,a):
    return np.array([a*np.ones_like(x)])

p1,_ = curve_fit(fitf,aconn_x,1.-qval_y,p0=[1./20.],method="lm",jac=fitf_jac)
p2,_ = curve_fit(fitf,act_conn_x,1.-qval_y,p0=[1./0.06],method="lm",jac=fitf_jac)
x1 = np.linspace(np.min(aconn),np.max(aconn),10)
x2 = np.linspace(np.min(act_conn),np.max(act_conn),10)
line1 = fitf(x1,p1[0])
line2 = fitf(x2,p2[0])

var = np.sum((qval_y-np.average(qval_y))**2)

r2_1 = 1-np.sum( (qval_y-fitf(aconn_x,p1[0]))**2 )/var
r2_2 = 1-np.sum( (qval_y-fitf(act_conn_x,p2[0]))**2 )/var
print(r2_1,r2_2)

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.plot(np.ravel(aconn),np.ravel(qvalues),'o')
#ax.plot(x1,1.-line1)
ax.set_ylim(maxqval+0.1,-0.05)
s = "Cumulative" if cumulative else "Average"
ax.set_xlabel(s+" synaptic contacts")
#ax.set_ylabel("1-Q value of observed connection")
ax.set_ylabel(y_label)
#ax.set_title("R^2 = 1 - e^2/sigma^2 = "+str(np.around(r2_1,3)))
fig.tight_layout()
if use_qvalues:
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/qvalues_vs_aconnectome.png",dpi=300,bbox_inches="tight")
elif use_pvalues:
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/pvalues_vs_aconnectome.png",dpi=300,bbox_inches="tight")

fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.plot(np.ravel(act_conn),np.ravel(qvalues),'o')
#ax.plot(x2,1.-line2)
ax.set_ylim(maxqval+0.1,-0.05)
ax.set_xlabel("simulated response based on connectome")
#ax.set_ylabel("1-Q value of observed connection")
ax.set_ylabel(y_label)
#ax.set_title("R^2 = 1 - e^2/sigma^2 = "+str(np.around(r2_2,3)))
fig.tight_layout()
folder = "/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/"
paper_folder = "/projects/LEIFER/francesco/funatlas/figures/paper/fig3/"
if use_qvalues:
    fig.savefig(folder+"qvalues_vs_act_connectome.png",dpi=300,bbox_inches="tight")
    if to_paper:
        fig.savefig(paper_folder+"qvalues_vs_act_connectome.pdf",dpi=300,bbox_inches="tight")
elif use_pvalues:
    fig.savefig(folder+"pvalues_vs_act_connectome.png",dpi=300,bbox_inches="tight")
    if to_paper:
        fig.savefig(paper_folder+"pvalues_vs_act_connectome.pdf",dpi=300,bbox_inches="tight")
plt.show()

