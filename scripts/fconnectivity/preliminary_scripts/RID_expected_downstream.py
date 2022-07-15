import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp

shuffle_connectome = False
shuffle_connectome_n = 1
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--shuffle-connectome": 
        shuffle_connectome = True
        shuffle_connectome_n = 1+int(sa[1])
    
plot = not shuffle_connectome_n > 1
if not plot: print("Skipping the plots because of the multiple shufflings.")

# DATASETS WITH STIMULATION OF RID
ds_list = [
    "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_101248/",
    "/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211102/pumpprobe_20211102_152524/",
    #"/projects/LEIFER/francesco/pumpprobe/AKSxneuropal/20211104/pumpprobe_20211104_102437/" #WT
    ]

funa = pp.Funatlas.from_datasets(
                ds_list,                
                merge_bilateral=True,merge_dorsoventral=False,merge_AWC=True)         

occ1, occ2 = funa.get_occurrence_matrix()
occ3 = funa.get_observation_matrix()    
ai_RID = funa.ids_to_i("RID")
funa.load_aconnectome_from_file(chem_th=1000,gap_th=0)
fr_inx = funa.get_fractional_gap_inx_mutants()
funa.aconn_gap *= fr_inx
funa.aconn_gap[funa.aconn_gap<0.5] = 0

scores = np.zeros(shuffle_connectome_n)
for i_shuffle in np.arange(shuffle_connectome_n):

    if shuffle_connectome and i_shuffle>0:
        # The 0-th iteration is always the non-shuffled one
        shuffling_sorter = funa.get_shuffling_sorter()
        funa.shuffle_aconnectome(shuffling_sorter)
        funa.shuffle_esconnectome(shuffling_sorter)

    fnames = ["external_data/GenesExpressing-npr-4-thrs2.csv",
              "external_data/GenesExpressing-npr-11-thrs2.csv",
              "external_data/GenesExpressing-pdfr-1-thrs2.csv",
              #"external_data/GenesExpressing-daf-2-thrs2.csv"
              ]
              
    trans_rec_pairs = ["FLP-14,NPR-4",
                       "FLP-14,NPR-11",
                       "PDF-1,PDFR-1",
                       #"INS-17,DAF-2"
                       ]
                       
    trans_exp_level = np.array([110634.0,110634.0,157972.0,])#1505.0])

    # Build expression levels                   
    x = np.arange(len(funa.head_ai))
    exp_levels = np.zeros((len(fnames),len(funa.head_ai)))
    for i_f in np.arange(len(fnames)):
        f = open(fnames[i_f],"r")
        lines = f.readlines()
        f.close()
        
        exp_levels_ = np.zeros(funa.n_neurons)
        
        
        for line in lines[1:]:
            s = line.split(",")
            cell_id = s[1][1:-1]
            exp_level = float(s[2])
            
            cell_id = funa.cengen_ids_conversion(cell_id)
            
            for cid in cell_id:
                ai = funa.ids_to_i(cid)
                if ai<0 and shuffle_connectome_n==0: print(cid,"not found")
                exp_levels_[ai] = exp_level*trans_exp_level[i_f]
            
        if shuffle_connectome and i_shuffle>0:
            exp_levels_ = funa.shuffle_array(exp_levels_,shuffling_sorter)
        b = funa.reduce_to_head(exp_levels_)
        exp_levels[i_f] = b
        
    # Sort them by the sum of the expression levels
    tot_exp_levels = np.sum(exp_levels,axis=0)
    bargsort = np.argsort(tot_exp_levels)[::-1]
    lim = np.where(tot_exp_levels[bargsort]==0)[0][0]
    #Print the last neuron to have a direct wireless connection to RID
    #print(funa.head_ids[bargsort][lim-1])
    # Now funa.head_ids[bargsort][:lim] are directly wirelessly connected to RID
    
    if plot:
        fig = plt.figure(1,figsize=(16,9))
        ax = fig.add_subplot(111)
        x = np.arange(len(funa.head_ai))

        prev_b = np.zeros(len(funa.head_ai))
        for i_f in np.arange(len(fnames)):
            b = exp_levels[i_f][bargsort]
            
            # Set to zero the neurons that are never observed
            for i_hn in np.arange(len(funa.head_ids)):
                id_hn = str(funa.head_ids[bargsort][i_hn])
                ai_hn = funa.ids_to_i(id_hn)
                if occ3[ai_hn,ai_RID]==0:
                    b[i_hn] = 0
            
            ax.bar(x,b,label=trans_rec_pairs[i_f],bottom=prev_b,alpha=0.5)
            prev_b += b

        #ax.set_xticks(x[:lim])
        ax.set_xticks(x)
        #ax.set_xticklabels(funa.head_ids[bargsort][:lim],rotation=90,fontsize=9)
        ax.set_xticklabels(funa.head_ids[bargsort],rotation=90,fontsize=9)
        #ax.set_xlim(-1,lim)
        ax.set_ylabel("CeNGEN transmitter count in RID * receptor counts in downstream neurons")
        ax.set_title("\"Expected\" amplitude of reponse based on receptor and transmitter \n"+\
                     "expression count from CeNGEN (transmitter,receptor)")
        ax.legend(loc=2)


    # USE THE ANATOMICAL AND THE KNOWN EXTRASYNAPTIC CONNECTOME TO DETERMINE WHICH
    # NEURONS, AMONG THE RESPONDING NEURONS, ARE 1 GAP JUNCTION AWAY FROM ONE OF
    # THE NEURONS DIRECTLY (EXTRASYNPATICALLY) CONNECTED TO RID.
    # THEN, LOOK AT THE ONES THAT ARE 1 WIRELESS CONNECTION AWAY FROM THE DIRECTLY
    # CONNECTED NEURONS.

    # FIRST, LABEL THE MEASURED RESPONSES THAT ARE SLOW AND FAST
    #slow_resp = ["RIA_","I4","SMBD_","I2_","IL2V_"]
    #intermediate_resp = ["AIM_","RIG_","RIV_","AVJ_","RME_","RMDD_"]
    #fast_resp = ["AWB_","AVA_","AIB_","RIH","AVE_","AVD_","M2_","M1","URYD_","CEPD_","OLQD_","RMDV_"]
    slow_resp = ["I4","I2_","IL2V_","IL1V_"]
    intermediate_resp = ["AIM_","RIG_","RIV_","AVJ_","RME_","RMDD_"]
    fast_resp = ["AWB_","AVA_","AIB_","RIH","AVE_","AVD_","M2_","M1","URYD_","CEPD_","OLQD_","RMDV_","M3_","I3"]

    # Get the neurons that show responses in the data.
    downst_in_data_ai = np.where(occ1[:,ai_RID]>0)[0]

    # Find single gap-junction hops
    one_gap_away = []
    for ai_resp in funa.head_ai:#downst_in_data_ai:
        # Skip the neurons that are already directly connected to RID
        if ai_resp in funa.head_ai[bargsort][:lim]: continue
        # Skip the neurons that are not in the dataset.
        if occ3[ai_resp,ai_RID]==0: continue
        # Iterate over the directly connected neurons
        for ai_direct_conn in funa.head_ai[bargsort][:lim]:
            if funa.aconn_gap[ai_resp,ai_direct_conn]>0 and occ1[ai_direct_conn,ai_RID]>0: #FIXME only from neurons that we respond in our datasets
                if not ai_resp in one_gap_away:
                    one_gap_away.append(ai_resp)
    one_gap_away = np.array(one_gap_away)
    if plot: print("one_gap_away",funa.neuron_ids[one_gap_away])
                
    # Find single extrasynaptic hops, excluding hops that have already a gap junction
    one_esyn_away = []
    esconn = funa.get_esconn()
    for ai_resp in funa.head_ai:#downst_in_data_ai:
        # Skip the neurons that are either directly connected to RID or are already
        # in one_gap_away.
        if ai_resp in funa.head_ai[bargsort][:lim] or ai_resp in one_gap_away: 
            continue
        # Skip the neurons that are not in the dataset.
        if occ3[ai_resp,ai_RID]==0: continue
        # Iterate over the neurons that are connected to RID or are in one_gap_away
        if len(funa.head_ai[bargsort][:lim])==0:
            joint_predicted_fast_responders = one_gap_away
        elif len(one_gap_away)==0:
            joint_predicted_fast_responders = funa.head_ai[bargsort][:lim]
        else:
            joint_predicted_fast_responders = np.append(funa.head_ai[bargsort][:lim],one_gap_away)
            
        for ai_direct_conn in joint_predicted_fast_responders:
            if esconn[ai_resp,ai_direct_conn]>0 and occ1[ai_direct_conn,ai_RID]>0: #FIXME only from neurons that we respond in our datasets
                if not ai_resp in one_esyn_away:
                    one_esyn_away.append(ai_resp)
                    
    one_esyn_away = np.array(one_esyn_away)
    if plot: print("one_esyn_away",funa.neuron_ids[one_esyn_away])

    # Convert the atlas indices to indices in the head
    downst_in_data_ai_head = funa.ai_to_head(downst_in_data_ai)

    exp_bars = np.zeros_like(tot_exp_levels)
    exp_bars[downst_in_data_ai_head] = occ1[downst_in_data_ai,ai_RID]/occ3[downst_in_data_ai,ai_RID]
    exp_bars = exp_bars[bargsort]
    new_lim = np.max(np.where(exp_bars!=0)[0])
    
    if plot:
        axb = ax.twinx()
        
    score = 0
    max_possible_score = 0
    h_ais_barg = funa.head_ai[bargsort]   
    found_slow_resp = False
    found_fast_resp = False
    found_intermediate_resp = False
    for i in np.arange(exp_bars.shape[0]):
        if h_ais_barg[i] in one_esyn_away:
            c = 'r'
        elif h_ais_barg[i] in one_gap_away:
            c = 'b'
        elif h_ais_barg[i] in funa.head_ai[bargsort][:lim]:
            c = 'b'
        else:
            c = 'g'
            
        lbl = ""
        if funa.neuron_ids[h_ais_barg[i]] in slow_resp:
            if not found_slow_resp:
                found_slow_resp = True
                lbl = "slow responses"
            mrkr = "o"
        elif funa.neuron_ids[h_ais_barg[i]] in intermediate_resp:
            mrkr = "*"
            if not found_intermediate_resp:
                found_intermediate_resp = True
                lbl = "intermediate responses"
        elif funa.neuron_ids[h_ais_barg[i]] in fast_resp:
            mrkr = "x"
            if not found_fast_resp:
                found_fast_resp = True
                lbl = "fast responses"
        else:
            mrkr = "^"
            
        # Use the markers and the colors as classifiers to calculate a score.
        # Add a point if the connectomic expectation (color) does not match
        # the data type (marker). The "correct" correspondences are:
        # b x
        # r o
        # while we don't know what to expect for either g or *. Include * in
        # the x for now.
        if ((c=="r" and (mrkr=="x" or mrkr=="*")) or (c=="b" and mrkr=="o") or (c=="g" and mrkr=="x")) and occ3[h_ais_barg[i],ai_RID]>0:
            score += 1
            
        if (not (mrkr=="^")) and occ3[h_ais_barg[i],ai_RID]>0: 
            max_possible_score += 1
        
            
        if occ3[h_ais_barg[i],ai_RID]>0:
        
            if plot:
                if lbl != "":
                    axb.plot((i,i),(-1,exp_bars[i]),c=c,marker=mrkr,label=lbl)
                else:
                    axb.plot((i,i),(-1,exp_bars[i]),c=c,marker=mrkr)
        
    if plot:
        axb.set_ylim(0,)            
        axb.set_ylabel("Fraction of times neurons responded in experiments")
        axb.legend(loc=1)

        fig.tight_layout()
        plt.show()
    
    scores[i_shuffle] = score

# If you are not plotting 
if shuffle_connectome_n>1:
    stdev = np.std(scores[1:])
    avg = np.average(scores[1:])
    print("Non-shuffled connectome is at "+str(np.around(abs(avg-scores[0])/stdev,2))+" sigmas")
    print("Best possible case is at "+str(np.around(avg/stdev,2))+" sigmas")
    
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    
    ax2.axvline(scores[0],c="k",label="not shuffled")
    ax2.hist(scores[1:],bins=15,label="shuffled",density=False)
    ax2.axvline(0,c="g",label="best possible case")
    ax2.axvline(max_possible_score,c="r",label="worst possible case")
    
    ax2.set_xlabel("number of wrong predictions")
    ax2.set_ylabel("distribution")
    ax2.set_title("Wrong predictions (true vs "+str(shuffle_connectome_n-1)+" shuffled connectomes)")
    ax2.legend(loc=1)
    plt.show()
    
