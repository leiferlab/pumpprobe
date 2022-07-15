import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm

plt.rc("xtick",labelsize=14)
plt.rc("ytick",labelsize=14)

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"

#ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "D20 E32 old mutant"
multi_conditions = "--multi-conditions" in sys.argv
drop_saturation_branches = "--drop-saturation-branches" in sys.argv or multi_conditions
take_winner = "--take-winner" in sys.argv
use_median = "--use-median" in sys.argv 
use_average = "--use-average" in sys.argv or not (take_winner or use_median)
q_th = 0.3
leq_rise_time = 0.55
chem_th = 0 
gap_th = 0
max_hops = 1
do_print = "--print" in sys.argv or not multi_conditions
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--ds-exclude-tags": 
        ds_exclude_tags=sa[1]
        if ds_exclude_tags == "None": ds_exclude_tags=None
    if sa[0] == "--ds-tags": ds_tags=sa[1]
    if sa[0] == "--leq-rise-time": leq_rise_time = float(sa[1])
    if sa[0] == "--chem-th": chem_th = float(sa[1])
    if sa[0] == "--gap-th": gap_th = float(sa[1])
    if sa[0] == "--max-hops": max_hops = int(sa[1])
    if sa[0] == "--q-th": q_th = float(sa[1])

print("#### merging bilaterally")
funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                verbose=False)
                
occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=True)
occ3 = funa.get_observation_matrix()

if q_th<1.0:
    occ1_inclall, occ2_inclall = funa.get_occurrence_matrix(req_auto_response=True,inclall=True)
    qvalues = funa.get_kolmogorov_smirnov_q(occ2_inclall)

    qvalues[np.isnan(qvalues)] = 1.
    for ii in np.arange(qvalues.shape[0]):
        for jj in np.arange(qvalues.shape[1]):
            if qvalues[ii,jj]>q_th:
                occ2[ii,jj] = []
    occ1 = funa.regenerate_occ1(occ2)

time1 = np.linspace(0,30,1000)
time2 = np.linspace(0,200,1000)

rise_times = funa.get_eff_rise_times(occ2,time2,True,drop_saturation_branches)
if use_average: 
    conf1, conf2 = funa.get_labels_confidences(occ2)
    dFF = funa.get_max_deltaFoverF(occ2,time1)
    avg_rise_times = funa.weighted_avg_occ2style2(rise_times,[dFF,conf2])
if use_median: med_rise_times = funa.median_occ2(rise_times)

act_conn = np.loadtxt("/projects/LEIFER/francesco/simulations/activity_connectome_sign2/activity_connectome_bilateral_merged.txt")
act_conn[np.isnan(act_conn)] = 0
sparseness = np.sum(qvalues<q_th)/np.prod(qvalues.shape)
th = pp.Funatlas.threshold_to_sparseness(act_conn,sparseness)
act_conn2 = np.abs(act_conn)>th

if multi_conditions:
    conditions = [{"leq_rise_time": 0.02,"gap_th":0.0,"chem_th":0.0},
                  {"leq_rise_time": 0.02,"gap_th":0.0,"chem_th":1e6},
                  {"leq_rise_time": 0.02,"gap_th":1e6,"chem_th":0.0}]
else:
    conditions = [{"leq_rise_time":leq_rise_time,"gap_th":gap_th,"chem_th":chem_th}]
    
aconnfast = []
aconnslow = []          
              
for cond in conditions:
    leq_rise_time = cond["leq_rise_time"]
    gap_th = cond["gap_th"]
    chem_th = cond["chem_th"]
    
    if take_winner:
        # For each connection, check if it has more fast or more slow kernels
        # and keep only the ones that are the majority.
        
        # Filter occ2 based on a fixed threshold on the rise times
        sysargv = ["--leq-rise-time:"+str(leq_rise_time),"--use-kernels"]#0.45
        if drop_saturation_branches: sysargv.append("--drop-saturation-branches")
        occ1fast,occ2fast,occ1slow,occ2slow = funa.filter_occ12_from_sysargv(occ2,sysargv,return_all=True)
        occ1fast,occ2fast,occ1slow,occ2slow = funa.take_winner(occ2fast,occ2slow)
        
        occ1fastbool = occ1fast>0
        occ1slowbool = occ1slow>0
        #occ1bool = occ1>0
        ##occ1slowbool = np.logical_and(occ1slow>0,~occ1fastbool)
        
    elif use_median:
        # For each connection, calculate the median rise time.
        occ1fastbool = med_rise_times<=leq_rise_time
        occ1slowbool = med_rise_times>leq_rise_time
        
    elif use_average:
        # For each connection, calculate the average rise time.
        occ1fastbool = avg_rise_times<=leq_rise_time
        occ1slowbool = avg_rise_times>leq_rise_time
        
    occ1bool = occ1fastbool+occ1fastbool
    
    if not do_print:
        print("n of fast kernels",np.sum(occ1fastbool))
        print("n of slow kernels",np.sum(occ1slowbool))
        
    # Load connectomes
    funa.load_aconnectome_from_file(chem_th=chem_th,gap_th=gap_th,exclude_white=False)
    aconn = funa.get_boolean_aconn()
    if max_hops>1:
        aconn = funa.get_effective_aconn3(max_hops=max_hops,gain_1=4)>1
    escon = funa.get_esconn()
    
    aconnfast_ = np.around(np.sum(aconn*occ1fastbool)/np.sum(occ1fastbool),2)
    aconnslow_ = np.around(np.sum(aconn*occ1slowbool)/np.sum(occ1slowbool),2)
    
    aconnfast.append(aconnfast_)
    aconnslow.append(aconnslow_)

    if do_print:
        print("--> sums of boolean-ized arrays = counts")
        print("##Total numbers")
        print("Number of functional connections considered",np.sum(occ1bool>0))
        print("Number of fast functional connections", np.sum(occ1fastbool))
        print("Number of slow functional connections", np.sum(occ1slowbool))
        print("") 
        print("##Fraction of all possible connections that are actually connected (connectomes)")
        print("What is the chance that if you pick two neurons, they are connected?")
        print("sum(in aconn)/maxconn",np.around(np.sum(aconn)/np.prod(aconn.shape),4))
        print("sum(in escon)/maxconn",np.around(np.sum(escon)/np.prod(escon.shape),4))
        print("")
        print("##Fraction of all possible connections that are actually connected (functionally)")
        print("sum(detected)/maxconn",np.around(np.sum(occ1bool)/np.prod(occ1bool.shape),2))
        print("")
        print("##Fractions of the intersections")
        print("sum(detected & in aconn)/sum(detected)",np.around(np.sum(occ1bool*aconn)/np.sum(occ1bool),2))
        print("sum(detected & not in aconn)/sum(detected)",np.around(np.sum(occ1bool*(~aconn))/np.sum(occ1bool),2))
        print("sum(detected & in escon)/sum(detected)",np.around(np.sum(occ1bool*escon)/np.sum(occ1bool),2))
        print("sum(detected & not in escon)/sum(detected)",np.around(np.sum(occ1bool*(~escon))/np.sum(occ1bool),2))
        print("sum(detected & in aconn or escon)/sum(detected)",np.around(np.sum(occ1bool*(aconn^escon))/np.sum(occ1bool),2))
        print("sum(detected & not aconn or escon)/sum(detected)",np.around(np.sum(occ1bool*(~(aconn^escon)))/np.sum(occ1bool),2))
        print("")
        print("Fast means kernels with rise times less than "+str(leq_rise_time)+" (drop_saturation_branches = "+str(drop_saturation_branches)+")")
        print("Chem th "+str(chem_th)+". Gap th "+str(gap_th))
        #print("sum(in aconn & fast)/sum(in aconn)",np.around(np.sum(aconn*occ1fastbool)/np.sum(aconn),2))
        print("sum(in aconn & fast)/sum(fast)",aconnfast_)
        #print("sum(in aconn & slow)/sum(in aconn)",np.around(np.sum(aconn*occ1slowbool)/np.sum(aconn),2))
        print("sum(in aconn & slow)/sum(slow)",aconnslow_)
        #print("sum(in escon & fast)/sum(in escon)",np.around(np.sum(escon*occ1fastbool)/np.sum(escon),2))
        print("sum(in escon & fast)/sum(fast)",np.around(np.sum(escon*occ1fastbool)/np.sum(occ1fastbool),2))
        #print("sum(in escon & slow)/sum(in escon)",np.around(np.sum(escon*occ1slowbool)/np.sum(escon),2))
        print("sum(in escon & slow)/sum(slow)",np.around(np.sum(escon*occ1slowbool)/np.sum(occ1slowbool),2))
        print("")
        
if multi_conditions:
    are_leq_rise_times_all_equal = True
    lrt0 = conditions[0]["leq_rise_time"]
    for c in conditions[1:]:
        if c["leq_rise_time"]!= lrt0: are_leq_rise_times_all_equal = False
    
    if not are_leq_rise_times_all_equal:
        print("Not all rise times are equal. Quitting.")
        quit()
    
    actconnfast_ = np.around(np.sum(act_conn2*occ1fastbool)/np.sum(occ1fastbool),2)
    actconnslow_ = np.around(np.sum(act_conn2*occ1slowbool)/np.sum(occ1slowbool),2)
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.bar((0,1),(actconnfast_,actconnslow_),color="C3",label="activity conn")
    ax.set_xticks([0,1])
    ax.set_xticklabels(["fast\n(tot. "+str(np.sum(occ1fastbool))+")","slow\n(tot. "+str(np.sum(occ1slowbool))+")"],rotation=45,ha="center",fontsize=14)
    ax.set_ylabel("fraction",fontsize=14)
    ax.legend()
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/fastslowanatomical3_qth_"+str(q_th)+"_bar.png",dpi=300,bbox_inches="tight")
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/fastslowanatomical3_qth_"+str(q_th)+"_bar.svg",bbox_inches="tight")
    
    fig3 = plt.figure(3,figsize=(4,6))
    ax3 = fig3.add_subplot(111)
    ax3.plot((0,1),(aconnfast[0],aconnslow[0]),'-o',label="all anatomical")
    ax3.plot((0,1),(aconnfast[1],aconnslow[1]),'-o',label="gap junctions")
    ax3.plot((0,1),(aconnfast[2],aconnslow[2]),'-o',label="chemical synapses")
    ax3.plot((0,1),(actconnfast_,actconnslow_),'-o',label="activity conn")
    ax3.set_xticks([0,1])
    ax3.set_xticklabels(["fast\n(tot. "+str(np.sum(occ1fastbool))+")","slow\n(tot. "+str(np.sum(occ1slowbool))+")"],rotation=45,ha="center",fontsize=14)
    #ax3.set_yticks([0.1,0.15,0.2,0.25])
    ax3.set_ylabel("fraction",fontsize=14)
    ax3.legend()
    ax3.set_title("q value threshold "+str(q_th))
    fig3.tight_layout()
    fig3.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/fastslowanatomical3_qth_"+str(q_th)+".png",dpi=300,bbox_inches="tight")
    fig3.savefig("/projects/LEIFER/francesco/funatlas/figures/compare_connectomes/fastslowanatomical3_qth_"+str(q_th)+".svg",bbox_inches="tight")
    plt.show()
