import numpy as np, os, pickle, gc
import pumpprobe as pp, mistofrutta as mf
from scipy.optimize import root,minimize,least_squares
import matplotlib.pyplot as plt

class Fconn:
    '''Class representing the functional connectome extracted from a dataset.'''
    
    nan_thresh = 0.05
    '''Global nan threshold'''

    folder = ""
    "Folder containing the dataset"
    n_neurons = 0
    "Number of neurons tracked in the recording"
    n_stim = 0
    "Number of stimulations applied during the recording"
    
    stim_neurons = np.zeros(n_stim,dtype=int)
    "Indices of the stimulated neurons (as matched to the reference volume)"
    stim_indices = np.zeros(n_stim,dtype=int)
    "Indices of the volumes at which the stimulations happened"
    resp_neurons = np.zeros((n_neurons,n_neurons))*np.nan
    
    filename = "fconn.pickle"
    "Default destination file name for the pickled object"
    
    filename_manually_located = "targets_manually_located.txt"
    "File containing the manually located targets."
    
    filename_flagged_responses = "flagged_responses.txt"
    "File containing the flagged responses. Each row is stim neu_i"
    
    filename_flagged_responses_add = "flagged_responses_add.txt"
    "File containing the flagged responses to be added. Each row is stim neu_i"
    
    filename_merged_reference_neurons = "merged_reference_neurons.txt"
    "File containing the merged reference neurons."
    
    filename_manually_located_comments = "targets_manually_located_comments.txt"
    "File containing the comments on the manually located targets."
    
    filename_bubble = "bubble.txt"
    "File containing the first volume with a bubble."
    
    def __init__(self, folder, n_neurons, n_stim):
        '''The constructor just sets up the arrays of the correct sizes.
        
        Parameters
        ----------
        folder: string
            Folder containing the recording.
        n_neurons: int
            Number of (tracked) neurons.
        n_stim: int
            Number of stimulations    
        '''
        
        if folder[-1]!="/": folder+="/"
        self.folder = folder
        self.n_neurons = n_neurons
        self.n_stim = n_stim
        self.verbose = True
        self.manually_located_present = False
        self.flagged_responses_present = False
        self.flagged_responses_add_present = False
        self.checked_merged_reference_neurons = False
        
        self.stim_neurons = -1*np.ones(n_stim,dtype=int)
        self.stim_neurons_compl_labels = np.array([None for i in np.arange(n_stim)])
        self.stim_indices = np.ones(n_stim,dtype=int)
        self.resp_neurons = np.nan*np.zeros((n_neurons,n_neurons))
        self.targeted_neuron_hit = np.zeros(n_stim,dtype=bool)
        
        self.resp_neurons_by_stim = [[] for q in np.arange(self.n_stim)]
        self.resp_ampl_by_stim = [[] for q in np.arange(self.n_stim)]
        
        self.peak_time = np.zeros((n_neurons,n_neurons))*np.nan
        
        # Store the volume indices used to cut the signal in the creation of
        # this fconn
        self.i0s = -1*np.ones(n_stim,dtype=int)
        self.i1s = -1*np.ones(n_stim,dtype=int)
        self.shift_vols = -1*np.ones(n_stim,dtype=int)
        self.next_stim_after_n_vol = -1*np.zeros(n_stim,dtype=int)
        
        # Store information about the fits
        self.fit_params_default = {"params": [], "n_branches": 0, "n_branch_params": []}
        self.fit_params_unc = [ [ self.fit_params_default.copy() for j in np.arange(n_neurons)] for i in np.arange(n_stim) ]
        self.fit_params = [ [ self.fit_params_default.copy() for j in np.arange(n_neurons)] for i in np.arange(n_stim) ]
        
        # Store the parameters used for the response detection
        self.nan_thresh = None
        self.deriv_thresh = None
        self.ampl_thresh = None
        self.deriv_min_time = None
        self.ampl_min_time = None
            
    @classmethod        
    def from_file(cls, folder, filename=None,verbose=True):
        '''Load the pickled version of a Fconn object.
        
        Parameters
        ----------
        folder: string
            Folder containing the recording.
        filename: string (optional)
            File name of the pickled object. If None, the default file name is 
            used. Default: None.
            
        Returns
        -------
        inst: Fconn 
            Instance of the class.
        '''
    
        if filename is None: 
            filename = cls.filename
        else: 
            if filename.split(".")[-1] != "pickle": filename += ".pickle"
        
        if os.path.isfile(folder+filename):
            f = open(folder+filename,"rb")
            inst = pickle.load(f)
            f.close()
            inst.verbose=verbose
            inst.load_manually_located()
            #inst.load_flagged_responses()
            bubble_stim = inst.get_bubble_stim()
            if bubble_stim is not None:
                inst.stim_neurons[bubble_stim:] = -2
            return inst
        else:
            print(folder+filename+" is not present.")
            quit()
            
    @classmethod
    def from_objects(cls, rec, brains, signal, 
                     delta_t_pre, 
                     nan_thresh=None, deriv_thresh=1.0, ampl_thresh=1.0,
                     deriv_min_time=2.0, ampl_min_time=5.,
                     matchless_autoresponse=False,
                     verbose=True):
        '''Creates an instance of a Fconn object from the recording, Brains, and
        Signal objects of the dataset.
        
        Parameters
        ----------
        rec: wormdatamodel.data.recording
            Recording object.
        brains: wormbrains.Brains
            Brains object.
        signal: wormdatamodel.signal.Signal
            Signal object.
        delta_t_pre: float
            Negative-time interval (in s) to consider before each stimulus.
            
        Returns
        -------
        inst: Fconn
            Instance of the class.
        
        '''
        if nan_thresh is None: nan_thresh = cls.nan_thresh
        events = rec.get_events()
        brains.load_matches(rec.folder)
        
        n_stim = events['optogenetics']['index'].shape[0]
        n_neurons = signal.data.shape[1]
        shift_vol = int(delta_t_pre/rec.Dt)
        
        inst = cls(rec.folder, n_neurons, n_stim)
        inst.shift_vol = shift_vol
        inst.Dt = rec.Dt
        inst.verbose = verbose
        
        print_warning_about_changes = True
        
        inst.nan_thresh = nan_thresh
        inst.deriv_thresh = deriv_thresh
        inst.ampl_thresh = ampl_thresh
        inst.deriv_min_time = deriv_min_time
        inst.ampl_min_time = ampl_min_time
        
        #############################
        # Find the stimulated neurons
        #############################
        int_btw_stim = 10
        for ie in np.arange(n_stim):
            ee = events['optogenetics']['index'][ie]
            delta_ee = 0
            if ee>=len(brains.nInVolume): #To save full-disk, truncated recordins.
                inst.stim_indices[ie] = ee
                continue
            while brains.nInVolume[ee] == 0: ee = ee - 1; delta_ee -= 1
            if delta_ee>0: print("Targeted brain empty, shifted by",str(delta_ee))
            target_coords = events['optogenetics']['properties']['target'][ie]
            #target_index = brains.get_closest_neuron(ee,target_coords,coord_ordering="xyz",z_true=True,inverse_match=False)
            target_index_ref = brains.get_closest_neuron(ee,target_coords,coord_ordering="xyz",z_true=True,inverse_match=True)
            '''
            ieshifts = 5
            target_index_refs = []
            for ieshift in np.arange(ieshifts):
                if brains.nInVolume[ee-ieshifts//2+ieshift] != 0:
                    bla = brains.get_closest_neuron(ee-ieshifts//2+ieshift,target_coords,coord_ordering="xyz",z_true=True)
                    if bla != -1: target_index_refs.append(bla)
            if len(target_index_refs) > 0:
                target_index_refs = np.array(target_index_refs)
                target_index_ref = np.unique(target_index_refs)[-1]
            else:
                target_index_ref = -1
            '''    
            inst.stim_indices[ie] = ee
            inst.stim_neurons[ie] = target_index_ref
        
        if inst.verbose: print("Fconn: Loading also manually located targets.")
        inst.load_manually_located()
        
        if inst.verbose: print("Fconn: Checking merged reference neurons.")
        inst.check_merged_reference_neurons()
        
        ################################
        # Automatically detect responses
        ################################
        
        set_shift_vol = shift_vol
        old_selected_ = np.zeros(signal.data.shape[1],dtype=bool)
        inst.nan_selection_autoresponse_bypassed = np.zeros(n_stim,dtype=bool)
        
        #Prepare second derivative
        #sderker = savgol_coeffs(13, 2, deriv=2, delta=inst.Dt)
        #sder_ = np.zeros_like(signal.data)            
        #for k in np.arange(signal.data.shape[1]):
        #    sder_[:,k] = np.convolve(sderker,signal.data[:,k],mode="same")
                                     
        for ie in np.arange(n_stim):
            ee = events['optogenetics']['index'][ie]
            target_index_ref = inst.stim_neurons[ie]
            
            # Find the indices that will be used to cut out a portion of the
            # signal array for this specific stimulation.
            i0 = max(0,ee-set_shift_vol) 
            shift_vol = set_shift_vol if i0>0 else ee
            inst.shift_vols[ie] = shift_vol
            i1 = min(int( i0 + 60/rec.Dt + shift_vol),signal.data.shape[0])
            inst.i0s[ie]=i0
            inst.i1s[ie]=i1
            if ie<n_stim-1:
                next_ee = events['optogenetics']['index'][ie+1]
                next_i0 = max(0,next_ee)
                vol_btw_stim = next_ee-ee
                int_btw_stim = vol_btw_stim*rec.Dt
                inst.next_stim_after_n_vol[ie] = vol_btw_stim
            else: inst.next_stim_after_n_vol[ie] = -1
            
            sig_seg = signal.get_segment(i0,i1,shift_vol,normalize="",baseline_mode="constant",baseline_range=[shift_vol//2,shift_vol])
            sig_seg_unsmoothed = signal.get_segment(i0,i1,shift_vol,normalize="",unsmoothed_data=True,baseline_mode="constant",baseline_range=[shift_vol//2,shift_vol])
            nan_mask = signal.get_segment_nan_mask(i0,i1)
            dr = signal.get_segment_derivative(i0,i1)
            #max_vol_n = int(min(40/rec.Dt,(int_btw_stim-5)/rec.Dt)) #-5 to avoid smoothing effects
            max_vol_n = int((int_btw_stim-5)/rec.Dt) #-5 to avoid smoothing effects
            
            ## Select responding neurons
            ##
            # OLD: noncontiguous condition
            # The traces should have a limited number of nans originally
            #nan_selection = np.sum(nan_mask,axis=0) < nan_thresh*(i1-i0)
            # The traces should not have contiguous nan regions longer than
            # a given fraction of the interval.
            nan_selection = np.zeros(n_neurons,dtype=bool)
            for i_neu in np.arange(n_neurons):
                nan_selection[i_neu] = cls.nan_ok(nan_mask[:,i_neu],nan_thresh*(i1-i0))
            
            # There needs to be a sufficient increase in the derivative
            # at least for a given amount of time
            #prev_dr = np.average(dr[shift_vol//2:shift_vol],axis=0) 
            prev_dr = np.average(np.absolute(dr[0:shift_vol-6]),axis=0) #shift_vol//2:shift_vol
            post_dr = dr[shift_vol:shift_vol+max_vol_n] 
            pre_avg_sig = np.median(signal.data[i0:i0+shift_vol],axis=0)
            deriv_min_vols = deriv_min_time/rec.Dt
            #deriv_selection = np.sum(np.absolute(post_dr)>deriv_thresh*prev_dr,axis=0)>=deriv_min_vols
            deriv_selection = np.sum(np.absolute(post_dr-prev_dr)/pre_avg_sig>deriv_thresh,axis=0)>=deriv_min_vols
            # You also want that at least a third of the points are before
            # 30 seconds, in case the max_vol_n is too large for this
            # recording. NEW 20211109
            if max_vol_n*rec.Dt>30:
                max_vol_n_p = int(30/rec.Dt)
                post_dr_p = dr[shift_vol:shift_vol+max_vol_n_p] 
                #deriv_selection *= np.sum(np.absolute(post_dr_p)>deriv_thresh*prev_dr,axis=0)>=(deriv_min_vols/3)
                deriv_selection *= np.sum(np.absolute(post_dr_p-prev_dr)/pre_avg_sig>deriv_thresh,axis=0)>=(deriv_min_vols/3)
                
            #sder_[i0+shift_vol-5:i0+shift_vol+16]
            
            '''
            # ADDITIONAL DERIVATIVE-BASED CRITERION. NOT CURRENTLY USED.
            # However, it needs to discard the cases in which the derivative is
            # large, but it's just decaying back to equilibrium from a previous
            # stimulation. 
            #prev_dr_avg = np.average(dr[:shift_vol],axis=0) #No abs!
            #post_dr_avg = np.average(post_dr,axis=0)
            #deriv_selection2 = ~(old_selected_ * (np.abs((post_dr_avg-prev_dr_avg)/prev_dr_avg) < 0.1)) #old_selected_
            ##print(deriv_selection2[19],np.abs((post_dr_avg-prev_dr_avg)/prev_dr_avg)[19],post_dr_avg[19],prev_dr_avg[19])'''
            '''##Old discard-criterion based on sign of derivative
            prev_dr_sign = np.sign(np.average(dr[shift_vol//2:shift_vol],axis=0))
            post_dr_sign = np.sign(np.average(post_dr,axis=0))
            deriv_selection2 = ~(old_selected_ * (prev_dr_sign == post_dr_sign))'''
            '''#deriv_selection *= deriv_selection2'''
            
            # And the signal needs to sufficiently rise above noise.
            # This includes three criteria. See below (CRITERIA ON AMPLITUDE
            # and check on SNR, but the SNR check is now deactivated.)
            
            r_restr = sig_seg[shift_vol:shift_vol+max_vol_n] 
            r_restr_unsmoothed = sig_seg_unsmoothed[shift_vol:shift_vol+max_vol_n]
            loc_std_seg = signal.get_loc_std(sig_seg[6:shift_vol-6],4)
            # Get the post-stimulation local std from the unsmoothed data
            loc_std_seg_post = signal.get_loc_std(r_restr_unsmoothed,4)
            glob_std_seg = np.std(r_restr,axis=0)
            ampl_min_vols = ampl_min_time/rec.Dt
            ampl_post = np.absolute(np.sum(r_restr,axis=0))
            
            # Two possibilities
            # FIRST CRITERION ON THE AMPLITUDE
            # 1) Either rise above (or below) a multiple of the absmax of the pre-stimulus signal, at least for a minimum amount of time.
            # This has to be signed, however. Otherwise traces will be selected in which the noise becomes larger but the average doesn't
            # move.
            # 1a: Go above. 1b: Go below.
            # 1_b: Go above/below with half the threshold but for twice the minimum contiguous time.
            # Former unsigned version: ampl_selection1 =  np.sum(np.abs(r_restr)>ampl_thresh*np.max(np.abs(sig_seg[:shift_vol]),axis=0),axis=0) > ampl_min_vols
            # First, remove spikes from the signal baseline interval.
            pre_sig = sig_seg[:shift_vol] # remove spikes
            absmax = np.sort(np.abs(pre_sig),axis=0)[-3]
            # The neuron needs to pass the threshold in a CONTIGUOUS number of volumes.
            # See https://stackoverflow.com/questions/24342047/count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array
            ampl_selection1a = np.zeros(n_neurons,dtype=bool)
            ampl_selection1a2 = np.zeros(n_neurons,dtype=bool)
            ampl_selection1b = np.zeros(n_neurons,dtype=bool)
            ampl_selection1b2 = np.zeros(n_neurons,dtype=bool)
            for i_neu in np.arange(n_neurons):
                # You need to check that an increase in spikes or noise
                # does not make the signal seem to go up. 
                #post_snr_ok = ampl_thresh*absmax[i_neu]>loc_std_seg_post[i_neu]
                post_snr_ok = True
                
                ###
                # Positive
                ###
                condition1a = r_restr[:,i_neu]>ampl_thresh*absmax[i_neu]
                if np.any(condition1a):
                    contcond1a = np.diff(np.where(np.concatenate(([condition1a[0]],
                                             condition1a[:-1] != condition1a[1:],
                                             [True])))[0])[::2]
                    ampl_selection1a[i_neu] = np.max(contcond1a)>ampl_min_vols
                else:
                    ampl_selection1a[i_neu] = False
                
                # Alternative version with half the threshold but twice the minimum time.
                condition1a2 = r_restr[:,i_neu]>0.5*ampl_thresh*absmax[i_neu]
                if np.any(condition1a2):
                    contcond1a2 = np.diff(np.where(np.concatenate(([condition1a2[0]],
                                             condition1a2[:-1] != condition1a2[1:],
                                             [True])))[0])[::2]
                    ampl_selection1a2[i_neu] = np.max(contcond1a2)>2*ampl_min_vols
                else:
                    ampl_selection1a2[i_neu] = False
                    
                # TODO Combine 1a and 1a2
                ampl_selection1a[i_neu] = ampl_selection1a[i_neu] or ampl_selection1a2[i_neu]
                    
                # You also want that at least a third of the points are before
                # 30 seconds, in case the max_vol_n is too large for this
                # recording. NEW 20211109
                if max_vol_n*rec.Dt>30:
                    max_vol_n_p = int(30/rec.Dt)
                    condition1a_p = r_restr[:max_vol_n_p,i_neu]>ampl_thresh*absmax[i_neu]
                    
                    ampl_selection1a[i_neu] *= np.sum(condition1a_p)>(ampl_min_vols/3)
                
                ###
                # Negative
                ###
                condition1b = r_restr[:,i_neu]<(-ampl_thresh*absmax[i_neu])
                if np.any(condition1b):
                    contcond1b = np.diff(np.where(np.concatenate(([condition1b[0]],
                                             condition1b[:-1] != condition1b[1:],
                                             [True])))[0])[::2]
                    ampl_selection1b[i_neu] = np.max(contcond1b)>ampl_min_vols  
                else:
                    ampl_selection1b[i_neu] = False
                
                # Alternative version with half the threshold but twice the minimum time.    
                condition1b2 = r_restr[:,i_neu]<(-0.5*ampl_thresh*absmax[i_neu])
                if np.any(condition1b2):
                    contcond1b2 = np.diff(np.where(np.concatenate(([condition1b2[0]],
                                             condition1b2[:-1] != condition1b2[1:],
                                             [True])))[0])[::2]
                    ampl_selection1b2[i_neu] = np.max(contcond1b2)>2*ampl_min_vols
                else:
                    ampl_selection1b2[i_neu] = False
                
                # TODO Combine 1b and 1b2    
                ampl_selection1b[i_neu] = ampl_selection1b[i_neu] or ampl_selection1b2[i_neu]
                    
                # You also want that at least a third of the points are before
                # 30 seconds, in case the max_vol_n is too large for this
                # recording. NEW 20211109
                if max_vol_n*rec.Dt>30:
                    max_vol_n_p = int(30/rec.Dt)
                    condition1b_p = r_restr[:max_vol_n_p,i_neu]<(-ampl_thresh*absmax[i_neu])
                    
                    ampl_selection1b[i_neu] *= np.sum(condition1b_p)>(ampl_min_vols/3)
                    
                if not post_snr_ok:
                    ampl_selection1a[i_neu] = False
                    ampl_selection1b[i_neu] = False
            
            # Old, noncontiguous versions
            #ampl_selection1a =  np.sum(r_restr>ampl_thresh*absmax,axis=0) > ampl_min_vols  
            #ampl_selection1b =  np.sum(r_restr<(-ampl_thresh*absmax),axis=0) > ampl_min_vols
            ampl_selection1 = np.logical_or(ampl_selection1a,ampl_selection1b)
            
            # SECOND CRITERION ON THE AMPLITUDE
            # 2) But the above selection would discard traces that have a large pre-stimulus signal because they were responding to the previous stimulation.
            # So, for the previously responding neurons (old_selected_) allow to rise even only above a multiple of the rolling standard deviation. The
            # neuron needs to pass the threshold in a CONTIGUOUS number of volumes.
            # 2*) But traces that pass this condition need to be subject to a further check, to avoid including /|\ in addition to /\|/\. See below.
            ## Old non-contiguous version, left as a reference to explain what is happening below.
            ##ampl_selection2 = (np.sum(np.abs(r_restr)>ampl_thresh*loc_std_seg,axis=0) > ampl_min_vols) * old_selected_
            ampl_selection2 = np.zeros(n_neurons,dtype=bool)
            for i_neu in np.arange(n_neurons):
                # Only for previously selected neurons
                if not old_selected_[i_neu]: continue 
                
                # Bypassing check on snr
                #post_snr_ok = ampl_thresh*loc_std_seg[i_neu]>loc_std_seg_post[i_neu]
                post_snr_ok = True
                
                condition2 = np.abs(r_restr[:,i_neu])>ampl_thresh*loc_std_seg[i_neu]
                if np.any(condition2) :
                    contcond2 = np.diff(np.where(np.concatenate(([condition2[0]],
                                             condition2[:-1] != condition2[1:],
                                             [True])))[0])[::2]
                    ampl_selection2[i_neu] = np.max(contcond2)>ampl_min_vols
                    
                    
                    if ampl_selection2[i_neu]:
                        # 2*) Only keep the response if the average derivative in the second half of the pre-segment (shift vol/2:shift_vol) is negative 
                        # (i.e. the sg kernel convolution gives a positive number) and the average derivative in the first half of this
                        # segment is positive (i.e. the sg kernel convolution is negative).
                        '''dr_2a = dr[shift_vol//2:shift_vol-6,i_neu]
                        dr_2b = dr[shift_vol:shift_vol+(i1-shift_vol)//3,i_neu]
                        if not (np.average(dr_2a)>=0 and np.average(dr_2b)<0):
                            ampl_selection2[i_neu] = False'''
                        
                        # Instead of using the sg-based derivative, do separate poly fits on the two parts 
                        #seg_2a = sig_seg_unsmoothed[shift_vol-25:shift_vol,i_neu]
                        seg_2a = sig_seg[shift_vol-25:shift_vol,i_neu]
                        pol_2a = np.polynomial.Polynomial.fit(np.arange(25),seg_2a,1)
                        len2b = min(25,len(sig_seg_unsmoothed)-shift_vol)
                        #seg_2b = sig_seg_unsmoothed[shift_vol:shift_vol+len2b,i_neu]
                        seg_2b = sig_seg[shift_vol:shift_vol+len2b,i_neu]
                        pol_2b = np.polynomial.Polynomial.fit(np.arange(len2b),seg_2b,1)
                        
                        dr_2a = pol_2a.deriv().linspace(25)[1][-1]
                        dr_2b = pol_2b.deriv().linspace(len2b)[1][0]
                        
                        if not (dr_2a<=5e-2 and dr_2b>=5e-2):
                            ampl_selection2[i_neu] = False
                            # If the response should be excluded based on this 
                            # criterion, then it should be excluded in general.
                            # Set also the other amplitude selection for this
                            # neuron to false. Note: This section of the code
                            # is not executed if the neuron had not been 
                            # detected at the previous stimulation.
                            ampl_selection1[i_neu] = False
                                                    
                        '''#diagnostics
                        if i_neu==3 and ie==6:
                            print(dr_2a,dr_2b)
                            print(ampl_selection1[i_neu],ampl_selection2[i_neu])
                            plt.plot(pol_2a.linspace(25)[1])
                            plt.plot(pol_2b.linspace(len2b)[1])
                            plt.plot(seg_2a)
                            plt.plot(seg_2b)
                            plt.show()'''
                            
                        
                    
                if not post_snr_ok:
                    ampl_selection2[i_neu] = False
                    
            ampl_selection = np.logical_or(ampl_selection1,ampl_selection2)
            # Old criterion
            #ampl_selection = np.sum(r_restr > ampl_thresh,axis=0)>ampl_min_vols #r is already normalized by the local standard deviation
            
            selected_ = deriv_selection*ampl_selection*nan_selection
            selected = np.where(selected_)[0]
            old_selected_ = selected_.copy()
            #if ie==0: print("Fconn: temporarily deactivated check on derivative")
            #selected = np.where(ampl_selection*nan_selection)[0]
            inst.resp_neurons_by_stim[ie] = selected
            inst.resp_ampl_by_stim[ie] = ampl_post[selected]
            
            if inst.stim_neurons[ie] not in [-2,-1]: 
                if np.isnan(inst.resp_neurons[0,target_index_ref]): 
                    inst.resp_neurons[:,target_index_ref] = 0
                inst.resp_neurons[selected,target_index_ref] = 1
                
        inst.targeted_neuron_hit = inst.has_targeted_neuron_been_hit(rec)

        bubble_stim = inst.get_bubble_stim()
        if bubble_stim is not None:
            inst.stim_neurons[bubble_stim:] = -2
        
        try:
            inst.find_peak_times(signal)
        except:
            pass
                
        return inst
            
    def to_file(self,folder=None,filename=None):
        '''Saves pickled version of the object to file.
        
        Parameters
        ----------
        folder: string (optional)
            Folder where to save the file. If None, the function uses the folder
            specified at instantiation. Default: None.
        filename: string (optional)
            Destination file name. If None, the function uses the default file
            name of the class. Default: None.
        '''        
        
        if folder is None:
            folder = self.folder
        
        if filename is not None:
            if filename.split(".")[-1] != "pickle": filename += ".pickle"
            self.filename = filename

        pickle_file = open(folder+self.filename,"wb")
        pickle.dump(self,pickle_file)
        pickle_file.close()
        
    def load_manually_located(self):
        self.manually_located_present = True
        fname = self.folder+self.filename_manually_located
        if os.path.isfile(fname):
            man_loc = np.loadtxt(fname,dtype=int)
            
            for m in np.arange(len(man_loc)):
                ml = man_loc[m]
                if ml!=-1: self.stim_neurons[m] = ml
        else:
            self.manually_located_present = False
            if self.verbose:
                print("Fconn: No file containing manually located targets "+\
                      "(should be "+self.filename_manually_located+")")
            
        fname = self.folder+self.filename_manually_located_comments
        if os.path.isfile(fname):
            f = open(fname,"r")
            lines = f.readlines()
            f.close()
            
            for i in np.arange(len(lines)):
                line = lines[i][:-1]
                ls = line.split(":")
                if ls[0]=="-3" and len(ls)>1:
                    self.stim_neurons[m] = -3
                    self.stim_neurons_compl_labels[i] = ls[1]
                    
    def load_flagged_responses(self):
        '''Load the file containinig the manually flagged responses. The file
        is written manually and each line must contain the stimulus index and
        the neuron (dataset) index separated by a space. These are best found
        by scrolling through the fits figure with each individual trace plotted
        separately.        
        '''
        fname = self.folder+self.filename_flagged_responses
        if os.path.isfile(fname):
            self.flagged_responses_present = True
            flagged_responses = np.loadtxt(fname,dtype=int,ndmin=2)
            
            for fr in flagged_responses:
                ifr=np.where(self.resp_neurons_by_stim[fr[0]]==fr[1])
                self.resp_neurons_by_stim[fr[0]] = \
                        np.delete(self.resp_neurons_by_stim[fr[0]],ifr)
            
        else:
            self.flagged_responses_present = False
            
        
        fname = self.folder+self.filename_flagged_responses_add
        if os.path.isfile(fname):
            self.flagged_responses_add_present = True
            flagged_responses_add = np.loadtxt(fname,dtype=int,ndmin=2)
            
            for fr in flagged_responses_add:
                if fr[1] not in self.resp_neurons_by_stim[fr[0]]:
                    print(fr[0],fr[1])
                    self.resp_neurons_by_stim[fr[0]] = \
                            np.append(self.resp_neurons_by_stim[fr[0]],fr[1])
            
        else:
            self.flagged_responses_add_present = False
                    
    def check_merged_reference_neurons(self):
        self.checked_merged_reference_neurons = True
        fname = self.folder+self.filename_merged_reference_neurons
        if os.path.isfile(fname):
            merged_neu = np.loadtxt(folder+"merged_ref_neurons.txt")
            merged_neu = np.where(merged_neu!=-1)[0]
            
            for sni in np.arange(len(self.stim_neurons)):
                if self.stim_neurons[sni] in merged_neu:
                    mni = np.where(self.stim_neurons[sni]==merged_neu)[0][0]
                    self.stim_neurons[sni] = merged_neu[mni]
                    
    def get_bubble_stim(self):
        fname = self.folder+self.filename_bubble
        if os.path.isfile(fname):
            bubble_stim = int(np.loadtxt(fname))
        else:
            bubble_stim = None
            
        return bubble_stim
            
                    
    def find_peak_times(self,signal):
        for ie in np.arange(self.n_stim)[1:]:
            i0 = self.i0s[ie]
            i1 = self.i1s[ie]
            if ie<self.n_stim-1:
                vol_btw_stim = self.i0s[ie+1]-i0
            else:
                vol_btw_stim = i1-i0
                
            sig_seg = signal.get_segment(i0,i1,self.shift_vol)
            
            # Look for maxima into the next stimulation segment
            # hoping for no overlaps.
            n_segs = 2
            assert i1>=self.shift_vol+vol_btw_stim
            selected = self.resp_neurons_by_stim[ie]
            target_index_ref = self.stim_neurons[ie]
            
            self.peak_time[selected,target_index_ref] = \
                  np.argmax(sig_seg[self.shift_vol:
                                    self.shift_vol+vol_btw_stim,
                                    selected],axis=0)*self.Dt
                                    
                                    
    def has_targeted_neuron_responded(self):
        has_responded = np.zeros(self.n_stim,dtype=bool)
        
        for ie in np.arange(self.n_stim):
            j = self.stim_neurons[ie] 
            has_responded[ie] = j in self.resp_neurons_by_stim[ie]
            
        return has_responded
    
    # VERSIONS WITH SIGNAL LOCALLY EXTRACTED FROM THE RECORDING OBJECT
    def has_targeted_neuron_been_hit(self,rec,
                                     n_vol=12,pre=10,
                                     include_detected_resp=False,
                                     return_all=False):
        '''Cross check that the targeting was successful. Combines 
        information about red photobleaching and green flashing.
        
        Parameters
        ----------
        rec: wormdm.data.recording
            Recording object.
        n_vol: int (optional)
            Total number of volumes from which to extract the signal. 
            Default: 12
        pre_vol: int (optional)
            Out of n_vol, how many volumes to take before the stimulus.
            Default: 10
        include_detected_resp: bool (optional)
            Whether to complement also with the automatically detected
            responses. Default: False.
        return_all: bool (optional)
            Whether to return also the red and green signals extracted by
            Fconn.extract_flash_photobl(). Default: False
            
        Returns
        -------
        has_been_hit: numpy.ndarray of bool
            Whether the crosscheck says the neuron was targeted, for each
            stimulation.
        red, green: numpy.ndarray
            Signal segments. Only returned if return_all is True.
        '''
         
        red,green = self.extract_flash_photobl(rec,n_vol,pre)                             
        has_photobleached = self.has_red_photobleached2(red,pre)
        has_flashed = self.has_green_flashed2(green,pre)
        has_been_hit = np.logical_or(has_photobleached,has_flashed)
        
        if include_detected_resp:
            has_responded = self.has_targeted_neuron_responded()
            has_been_hit = np.logical_or(has_been_hit,has_responded)
        
        if return_all:
            return has_been_hit,red,green
        else:
            return has_been_hit
    
    @staticmethod
    def extract_flash_photobl(rec,n_vol=12,pre_vol=10):
        '''Extracts the time-local image intensity at the position and time
        of the stimulations.
        
        Parameters
        ----------
        rec: wormdm.data.recording
            Recording object.
        n_vol: int (optional)
            Total number of volumes from which to extract the signal.
            Default: 12.
        pre_vol: int (optional)
            Out of n_vol, how many volumes to take before the stimulus.
            Default: 10.
        
        Returns
        -------
        red, green: 2D numpy.ndarray
            Red and green signals. Indexed as [stim_index, vol].
        '''
        events = rec.get_events()
        n_stim = events['optogenetics']['index'].shape[0]

        n_vol = 12
        pre_vol = 10
        green = np.zeros((n_stim,n_vol))
        red = np.zeros((n_stim,n_vol))
        for ie in np.arange(n_stim):
            ee = events['optogenetics']['index'][ie]
            target_coords = events['optogenetics']['properties']['target'][ie]
            xr,yr = int(target_coords[0]), int(target_coords[1])
            zyxr = np.array([[-1,yr,xr]])
            #_,yg,xg = wormdm.data.redToGreen(zyxr, folder=rec.folder)[0]
            _,yg,xg = rec.red_to_green(zyxr)[0]
            
            start_vol = ee-pre_vol
            rec.load(startVolume=start_vol,nVolume=n_vol)
            volFrame0 = rec.volumeFirstFrame[start_vol:start_vol+n_vol+1].copy()
            volFrame0 -= rec.volumeFirstFrame[start_vol]
            
            for i_vol in np.arange(n_vol):
                # Extract the portion of the green frame around the targeted position
                # Get the z in this current volume
                z = np.argmin(np.abs(rec.ZZ[start_vol+i_vol] - target_coords[2]))
                zp = volFrame0[i_vol] + z
                green[ie,i_vol] = np.sum(rec.frame[zp,1,yg-5:yg+6,xg-5:xg+6])
                red[ie,i_vol] = np.sum(rec.frame[zp,0,yr-5:yr+6,xr-5:xr+6])
                
        return red, green
        
    def has_red_photobleached2(self,red,pre=10,std_mult=1.5,return_all=False):
        '''Checks if the targeted neuron photobleached upon stimulation.
        
        Parameters
        ----------
        red: 2D numpy.ndarray
            Red signal. Indexed as [stim_i,vol]
        pre: int (optional)
            red[stim_i,:pre] are before the stimulus. Same as in 
            Fconn.extract_flash_photobl(). Default: 10
        std_mult: float (optional)
            Multiple of the standard deviation by which the neuron has to 
            photobleach. Default: 1.5.
        return_all: bool (optional)
            Whether to return also the signal segments. Default: False.
        
        Returns
        -------
        has_photobleached: numpy.ndarray of bool
            Whether the targeted neurons have photobleached.
        segs: list of numpy.ndarray
            Signal segments.
        '''
            
        has_photobleached = np.zeros(self.n_stim,dtype=bool)
        
        segs = []
        
        for ie in np.arange(self.n_stim):
            y = red[ie,:pre]
            ypost = red[ie,pre:]
            
            segs.append(red[ie])
            
            std = np.std(y)
            avg = np.average(y)
            avg_post = np.average(ypost)
            
            has_photobleached[ie] = avg-avg_post>std_mult*std
        
        if return_all:
            return has_photobleached, segs
        else:
            return has_photobleached  
            
    def has_green_flashed2(self,green,pre=10,std_mult=1.5,return_all=False):
        '''Checks if the targeted neuron flashed upon stimulation.
        
        Parameters
        ----------
        gree: 2D numpy.ndarray
            Green signal. Indexed as [stim_i,vol]
        pre: int (optional)
            green[stim_i,:pre] are before the stimulus. Same as in 
            Fconn.extract_flash_photobl(). Default: 10
        std_mult: float (optional)
            Multiple of the standard deviation by which the neuron has to 
            flash. Default: 3.0.
        return_all: bool (optional)
            Whether to return also the signal segments. Default: False.
        
        Returns
        -------
        has_flashed: numpy.ndarray of bool
            Whether the targeted neurons have flashed.
        segs: list of numpy.ndarray
            Signal segments.
        '''
        has_flashed = np.zeros(self.n_stim,dtype=bool)
        
        segs = []
        
        for ie in np.arange(self.n_stim):
            y = green[ie,:pre]
            yflash = green[ie,pre:]
            
            segs.append(green[ie])
            
            std = np.std(y)
            avg = np.average(y)
            flash = np.max(yflash)
            
            has_flashed[ie] = flash>(avg+std_mult*std)
        
        if return_all:
            return has_flashed, segs
        else:
            return has_flashed
            
    # OLD VERSIONS WITH THE SIGNAL OBJECTS
    def has_targeted_neuron_been_hit_sig(self,redsig,greensig,
                                     prepost=20,include_detected_resp=True):
        has_photobleached = self.has_red_photobleached(redsig,prepost)
        has_flashed = self.has_green_flashed(greensig,prepost)
        has_been_hit = np.logical_or(has_photobleached,has_flashed)
        
        if include_detected_resp:
            has_responded = self.has_targeted_neuron_responded()
            has_been_hit = np.logical_or(has_been_hit,has_responded)
        
        return has_been_hit
                                    
    def has_red_photobleached_sig(self,redsig,prepost=20,return_all=False):
        has_photobleached = np.zeros(self.n_stim,dtype=bool)
        
        segs = []
        
        for ie in np.arange(self.n_stim):
            i0 = self.i0s[ie]
            shift_vol = self.shift_vols[ie]
            j = self.stim_neurons[ie]
            
            ia = i0+shift_vol-prepost
            ib = i0+shift_vol+prepost
            
            ya = redsig[ia:ia+prepost,j]
            yb = redsig[ia+prepost:ib,j]
            
            segs.append(redsig[ia:ib,j])
            
            stda = np.std(ya)
            stdb = np.std(yb)
            avga = np.average(ya)
            avgb = np.average(yb)
            
            has_photobleached[ie] = avga-avgb>0.5*(stda+stdb)
        
        if return_all:
            return has_photobleached, segs
        else:
            return has_photobleached  
            
    def has_green_flashed_sig(self,greensig,prepost=20,return_all=False):
        has_flashed = np.zeros(self.n_stim,dtype=bool)
        
        segs = []
        
        for ie in np.arange(self.n_stim):
            i0 = self.i0s[ie]
            shift_vol = self.shift_vols[ie]
            j = self.stim_neurons[ie]
            
            ia = i0+shift_vol-prepost
            ib = i0+shift_vol+prepost
            
            y = greensig[ia:ia+prepost-1,j]
            #yb = greensig[ia+prepost+3:ib,j]
            #y = np.append(ya,yb)
            
            yflash = greensig[ia+prepost-1:ia+prepost+3,j]
                        
            segs.append(greensig[ia:ib,j])
            
            std = np.std(y)
            avg = np.average(y)
            flash = np.average(yflash)
            
            has_flashed[ie] = flash>(avg+1.5*std)
        
        if return_all:
            return has_flashed, segs
        else:
            return has_flashed
    
                                    
    ##################################
    # RESPONDING/NONRESPONDING NEURONS
    ##################################
    
    ##################################
    # METHODS REGARDING FIT PARAMETERS
    ##################################
                                    
    def clear_fit_results(self,stim,neu,mode="unconstrained"):
        if mode=="unconstrained":
            self.fit_params_unc[stim][neu] = self.fit_params_default.copy()
        elif mode=="constrained":
            self.fit_params[stim][neu] = self.fit_params_default.copy()
    
    @staticmethod    
    def get_irrarray_from_params(params):
        p = mf.struct.irrarray(params["params"],
                              [params["n_branch_params"]],["branch"])
                              
        return p
    
    def get_param_irrarray(self,stim,neu,constrained=False):
        if constrained:
            params_dict = self.fit_params[stim][neu]
        else:
            params_dict = self.fit_params_unc[stim][neu]
        
        irrarr = self.get_irrarray_from_params(params_dict)
        return irrarr
        
    def get_kernel(self,time,stim,neu):
        p = self.get_param_irrarray(stim,neu,constrained=True)
        ker = self.eci(time,p)
        
        return ker
        
    def get_kernel_ec(self,stim,neu):
        p = self.fit_params[stim][neu]
        if p["n_branches"]>0 and not p["params"][0]==0 and not p["params"][1]==1:#==[0,1]:
            ec = pp.ExponentialConvolution.from_dict(p)
        else:
            ec = None
        
        return ec
        
    def get_unc_fit_ec(self,stim,neu):
        '''Returns the ExponentialConvolution object of the unconstrained fit
        of a response.
        
        Parameters
        ----------
        stim: int
            Stimulation index.
        neu: int
            Neuron index.
        
        Returns
        -------
        ec: ExponentialConvolution
            Unconstrained fit of the response of neuron neu to stimulation stim.
            If the neuron did not respond, None is returned.
            
        '''
        p = self.fit_params_unc[stim][neu]
        if p["n_branches"]>0:
            ec = pp.ExponentialConvolution.from_dict(p)
        else:
            ec = None
        
        return ec
        
    @staticmethod    
    def get_gammas_from_params(params):
        p = np.copy(params["params"])
        tbdel = np.append(0,params["n_branch_params"])
        gammas = np.delete(p,tbdel)
        
        return gammas
        
    def get_gammas_by_stim(self,stim,constrained=True):
        if constrained:
            ps = self.fit_params[stim]
        else:
            ps = self.ft_params_unc[stim]
            
        gammass = []
        for p in ps:
            if p["n_branches"]!=0:
                gammass.append(self.get_gammas_from_params(p))
            else:
                gammass.append([])
        return gammass
    
    @staticmethod
    def get_branch_ampl_from_params(params):
        ampls = [params["params"][0]]
        q = 0
        for i in np.arange(len(params["n_branch_params"])-1):
            q += params["n_branch_params"]
            ampls.append(params["params"][q])
            
        return ampls
        
        
    def get_branch_ampl_by_stim(self,stim,constrained=True):
        if constrained:
            ps = self.fit_params[stim]
        else:
            ps = self.ft_params_unc[stim]
            
        amplss = []
        for p in ps:
            if p["n_branches"]!=0:
                amplss.append(self.get_branch_ampl_from_params(p))
            else:
                amplss.append([])
        return amplss
            
    def get_fit_signs(self,stim,neu):
        # Get reference to function
        
        sign = None
        print("Re-implement sign given generalized eci.")
        return sign
        
        
    def get_minmax_gammas(self,constrained=True):
        '''Returns the matrices containing the minimum and maximum gammas 
        (maximum and minimum taus) in the fitted kernels. The maximum gamma can 
        be indicative of whether there is an (effective) anatomical connection 
        between neurons or not.
        
        Parameters
        ----------
        constrained: bool (optional)
            Whether to use the constrained fits (i.e. of the actual kernels) or
            the unconstrained ones (i.e. the bare fit of the signal in neuron
            i). Default: True.
            
        Returns
        -------
        min_gammas: 2D array_like of floats
            min_gammas[i,j] is the minimum gamma fitted for the kernel from j
            to i. If there are multiple stimulations of neuron j, the minimum
            gamma among those stimulations is returned.
        max_gammas: 2D array_like of floats
            max_gammas[i,j] is the maximum gamma fitted for the kernel from j
            to i. If there are multiple stimulations of neuron j, the maximum
            gamma among those stimulations is returned.
        '''
        
        max_gammas = np.zeros((self.n_neurons,self.n_neurons))
        min_gammas = np.zeros((self.n_neurons,self.n_neurons))
        
        for ie in np.arange(self.n_stim):
            neu_j = self.stim_neurons[ie]
            neu_is = self.resp_neurons_by_stim[ie]
            
            gammass = self.get_gammas_by_stim(ie,constrained=constrained)

            for i in np.arange(len(neu_is)):
                neu_i = neu_is[i]
                if len(gammass[neu_i])!=0:
                    maxg = np.max(gammass[neu_i])
                    ming = np.min(gammass[neu_i])
                    # Check if that max_gamma had been already set. If so, only
                    # overwrite it if the new max_gamma is larger than the old 
                    # one.
                    if max_gammas[neu_i,neu_j]==0:
                        max_gammas[neu_i,neu_j] = maxg
                    else:
                        if max_gammas[neu_i,neu_j] < maxg:
                            max_gammas[neu_i,neu_j] = maxg
                            
                    if min_gammas[neu_i,neu_j]==0:
                        min_gammas[neu_i,neu_j] = ming
                    else:
                        if min_gammas[neu_i,neu_j] > ming:
                            min_gammas[neu_i,neu_j] = ming
        
        return min_gammas, max_gammas
        
            
    def is_peak_within(self, f, stim, neu=None, t=30., nt=100):
        if neu is None:
            neu = self.resp_neurons_by_stim[stim]
            neu_was_scalar = False
        else:
            try: len(neu); neu_was_scalar = False
            except: neu = [neu]; neu_was_scalar = True
            
        # Make time axis
        x = np.linspace(0.,t,nt)
        
        within = np.zeros(len(neu),dtype=np.bool)
        for i in np.arange(len(neu)):
            j = neu[i]
            p = self.get_irrarray_from_params(self.fit_params_unc[stim][j])
            y = np.absolute(self.eci(x,p))
            within[i] = np.argmax(y)<(x.shape[0]-1)
        
        if neu_was_scalar:
            return within[0]
        else:
            return within
        
            
    def get_stim_argsort(self):
        '''Get an argsort-type ordering array that will sort the neurons to make
        nicer colormap plots.
        
        Returns
        -------
        new_i: numpy.ndarray of int
            new_i[j] is the new index of neuron j.
        argsort_i: numpy.ndarray of int
            argsort_i is the "inverse" of new_i, to be used as an actual 
            argsort.
        '''
        new_i = np.zeros(self.n_neurons,dtype=np.int32)
        argsort_i = np.zeros(self.n_neurons,dtype=np.int32)
        found = np.zeros(self.n_neurons,dtype=np.bool)
        J = 0
        for j in np.arange(self.n_stim):
            targ_i = self.stim_neurons[j]
            if not found[targ_i] and targ_i not in [-2,-1]:
                new_i[targ_i] = J    # quando faccio indexing dei dati devo mettere in J ciò che corrisponde a targ_i --> W[new_i[targ_i]]
                argsort_i[J] = targ_i
                found[targ_i] = True
                J += 1
        n_targeted = J 
        for k in np.arange(len(found)):
            if not found[k]:
                new_i[k] = J
                argsort_i[J] = k
                J += 1
        
        return new_i, argsort_i
        
    def get_time_axis(self,ie,return_all=True):
        i0 = max(0,self.i0s[ie])
        i1 = self.i1s[ie]
        time = (np.arange(i1-i0)-self.shift_vols[ie])*self.Dt
        time2 = time[time>=0]
        
        if return_all: 
            return time, time2, i0, i1
        else:
            return time
        
                
    #####
    #####
    # ec2
    #####
    #####
    @staticmethod
    def _exp_conv_2(t,p):
        '''Convolution from 0 to t of e^-t/tau1 and e^-t/tau2.'''
        A = p[0]
        tau1 = p[1]
        tau2 = p[2]
        
        y = A * tau1*tau2/(tau1-tau2) * (np.exp(-t/tau1) - np.exp(-t/tau2))

        return y 
        
    ec2 = _exp_conv_2
                                    
    @classmethod
    def guess_p_ec2(cls,y,dt,p0=[2.,1.,0.5]):
        '''Given data to be fitted with _exp_conv_2, estimate the guess 
        parameters for the fit.        
        '''
        max_t = np.argmax(y)*dt
        max_val = np.max(y)
        integral = pp.integral(y,dt,8)
        
        sol = root(cls._guess_p_ec2_eqs,p0,args=(integral,max_t,max_val))
        
        return sol.x
        
        
    @staticmethod
    def _guess_p_ec2_eqs(p,integral,max_t,max_val):
        '''Given the integral, the peak time, and the peak value of 
        _exp_conv_2, make an estimate of the parameters of that function.
        
        They are
        A = integral/tau1/tau2 
        tau1 = tau2*np.exp(-max_t*(1./tau1 - 1./tau2))
        tau2 = max_val*(tau1-tau2)/A/tau1/(np.exp(-max_t/tau1)-np.exp(-max_t/tau2))
        '''
        a, tau1, tau2 = p
        f = [a-integral/tau1/tau2,
             tau2*np.exp(-max_t*(1./tau1 - 1./tau2)) - tau1,
             max_val*(tau1-tau2)/a/tau1/(np.exp(-max_t/tau1)-np.exp(-max_t/tau2))-tau2 ]
        return f
        
    @classmethod
    def fit_ec2(cls,x,y,beta0=0.1,avg_lgst_n=10,method=None):
        dt = x[1]-x[0]
        p0 = cls.guess_p_ec2(y,dt)
        
        res = minimize(cls.error,p0,args=(x,y,cls._exp_conv_2),method=method)
        p = res.x
        
        return p
        
    ######
    ######
    # ec2b
    ######
    ######
    @staticmethod
    def _exp_conv_2b(t,p):
        '''Same as _exp_conv_2b, but with alpha = gamma1 and 
        beta = gamma2-gamma1, where gamma1=1/tau1 and gamma2=1/tau2'''
        A, alpha, beta = p
        #alpha = np.abs(alpha) #TODO CONSTRAINTS
        #beta = max(-alpha,beta) #TODO CONSTRAINTS
        
        y = A/beta*np.exp(-alpha*t)*(1.0-np.exp(-beta*t))
        return y
        
    ec2b = _exp_conv_2b
    
    @staticmethod
    def _exp_conv_2b_jac(p,*args):
        A, alpha, beta = p
        alpha = np.abs(alpha) 
        t,y,f = args
        
        j1 = 1./beta*np.exp(-alpha*t)*(1.0-np.exp(-beta*t))
        j2 = -alpha*A/beta*np.exp(-alpha*t)*(1.0-np.exp(-beta*t))
        j3 = -A/np.power(beta,2)*np.exp(-alpha*t)*(1.0-np.exp(-beta*t))+\
             A*np.exp(-alpha*t)*np.exp(-beta*t)
             
        return np.sum(np.power(j1,2)),np.sum(np.power(j2,2)),np.sum(np.power(j3,2))
        
    @classmethod
    def guess_p_ec2b(cls,y,dt,beta0,avg_lgst_n=1):
        '''Given data to be fitted with _exp_conv_2b, estimate the guess 
        parameters for the fit.        
        '''
        max_t = np.argmax(np.abs(y))*dt
        sign = np.sign(y[int(max_t/dt)])
        max_val = np.average(np.sort(np.abs(y))[-avg_lgst_n:])
        integral = pp.integral(y,dt,8)
        
        max_val *= sign
        
        sol = root(cls._guess_p_ec2b_eqs,[beta0],args=(integral,max_t,max_val))
        beta = sol.x[0]
        alpha = beta/(np.exp(beta*max_t)-1.)
        A = alpha*(alpha+beta)*integral
        
        return [A,alpha,beta]
        
    @staticmethod
    def _guess_p_ec2b_eqs(p,integral,max_t,max_val):
        beta = p[0]
        alpha = beta/(np.exp(beta*max_t)-1.)
        A = alpha*(alpha+beta)*integral
        
        y = max_val - A/beta*np.exp(-alpha*max_t)*(1.-np.exp(-beta*max_t))
        
        return y
    
    @classmethod
    def fit_ec2b(cls,x,y,beta0=0.1,avg_lgst_n=10,method=None):
        dt = x[1]-x[0]
        p0 = cls.guess_p_ec2b(y,dt,beta0,avg_lgst_n)
        
        res = minimize(cls.error,p0,args=(x,y,cls._exp_conv_2b),method=method)
        p = res.x
        
        return p
    
    ######
    ######
    # ec3b
    ######
    ######    
    @staticmethod
    def _exp_conv_3b(t,p):
        '''Returns the following convolution
        C/(g2-g1) [exp(-g1 t) - exp(-g2 t)] * exp(-g3 t)
        '''
        A, g3, g13, g23 = p
        
        g3 = np.abs(g3) #TODO CONSTRAINTS AS ACTUAL CONSTRAINTS
        g13 = max(-g3,g13) #FIXME
        g23 = max(-g3,g23) #FIXME
        
        y = A * np.exp(-g3*t) * \
            ( (1.-np.exp(-g13*t))/g13 - (1.-np.exp(-g23*t))/g23 )
            
        return y
    
    ec3b = _exp_conv_3b
    
    @staticmethod     
    def _exp_conv_3b_jac(p,*args):
        '''Returns the following convolution
        C/(g2-g1) [exp(-g1 t) - exp(-g2 t)] * exp(-g3 t)
        '''
        A, g3, g13, g23 = p
        
        g3 = np.abs(g3) #TODO CONSTRAINTS AS ACTUAL CONSTRAINTS
        g13 = max(-g3,g13)
        g23 = max(-g3,g23)
        t,y,f = args
        
        j1 = np.exp(-g3*t) * \
            ( (1.-np.exp(-g13*t))/g13 - (1.-np.exp(-g23*t))/g23 )
        
        j2 = -g3*A * np.exp(-g3*t) * \
            ( (1.-np.exp(-g13*t))/g13 - (1.-np.exp(-g23*t))/g23 )
        j3 = A * np.exp(-g3*t) /np.power(g13,2) * ( (1.-np.exp(-g13*t)) - g13*g13*np.exp(-g13*t))
        j4 = -A * np.exp(-g3*t) /np.power(g23,2) * ( (1.-np.exp(-g23*t)) - g23*g23*np.exp(-g23*t))
        
        return np.sum(np.power(j1,2)),np.sum(np.power(j2,2)),np.sum(np.power(j3,2)),np.sum(np.power(j4,2))
        
    @classmethod
    def fit_ec3b(cls,x,y,p0=None,method=None):
        if p0 is None: p0 = [1.,0.3,0.01,0.01]
                        
        res = minimize(cls.error,p0,args=(x,y,cls._exp_conv_3b),method=method)
        p = res.x
        
        return p        
    
    ######
    ######
    # ec4b
    ######
    ######    
    @staticmethod
    def _exp_conv_4b(t,p):
        '''Returns the following convolution
        A 1/(g2-g1)[exp(-g1 t)-exp(-g2 t)]*1/(g3-g4)[exp(-g3 t)-exp(-g4 t)]
        '''
        A, g1, g2, g31, g41 = p
        
        g32 = g31+g1-g2
        g42 = g41+g1-g2
        g43 = g41-g31
        g21 = g2-g1
        
        #g1 = np.abs(g1) #TODO CONSTRAINTS AS ACTUAL CONSTRAINTS
        #g2 = np.abs(g2)
        #g31 = max(-g1,g31)
        #g41 = max(-g1,g41)
        #g32 = max(-g1,g32)
        #g42 = max(-g1,g42)
        
        y = A/g21/g43 * (np.exp(-g1*t) * ( \
                           (1.-np.exp(-g31*t))/g31 - (1.-np.exp(-g41*t))/g41)+\
                         np.exp(-g2*t) * ( \
                          -(1.-np.exp(-g32*t))/g32 + (1.-np.exp(-g42*t))/g42))
                          
        return y
        
    ec4b = _exp_conv_4b
    
    @staticmethod
    def get_rf_from_ec4b_p(x,p,g1,g2):
        '''From the parameters fit for ec4b, get the underlying response
        function.'''
        A, g1, g2, g31, g41 = p
        
        g3 = g31+g1
        g4 = g41+g1
        
        rf = A/(g4-g3)*(np.exp(-g3*x)-np.exp(-g4*x))
        return rf
        
    @classmethod
    def fit_ec4b(cls,x,y,p0=None,method=None,routine="least_squares",g1=None,g2=None):
        # Some default guesses
        if p0 is None: p0 = [1.,0.3,0.4,0.01,0.02]
        # Null constraints, except on g1 and g2, which have to be positive
        lower_bounds = [-np.inf,0.,0.,-np.inf,-np.inf]
        upper_bounds = [np.inf,np.inf,np.inf,np.inf,np.inf]
        # If g1 or g2 are not None, they are the constrained timescales for the
        # input.
        if g1 is not None:
            p0[1] = g1
            lower_bounds[1] = g1*0.9
            upper_bounds[1] = g1*1.1
        if g2 is not None:
            p0[2] = g2
            lower_bounds[2] = g2*0.9
            upper_bounds[2] = g2*1.1
        
        if routine == "minimize":
            res = minimize(cls.error,p0,args=(x,y,cls._exp_conv_4b),method=method)
            p = res.x
        elif routine == "least_squares":
            residuals = lambda p,x,y: cls.ec4b(x,p) - y
            res = least_squares(residuals,p0,args=(x,y),method=method,bounds=(lower_bounds,upper_bounds))
            p = res.x
            
        
        return p 
        
    ######
    ######
    # ecib
    ######
    ######   
    
    @staticmethod
    def eci(x,p,power_t=None):
        '''Evaluate 
        '''
        # 
        if not isinstance(p,mf.struct.irrarray):
            if power_t is None:
                ecj = pp.ExponentialConvolution(p[1:],p[0])
            else:
                # power_t tells you the power of the power of t in terms that
                # are like t^n exp(-gt)
                ecj = pp.ExponentialConvolution(p[1],p[0])
                if power_t[1] > 0:
                    for q in np.arange(power_t[1]): ecj.convolve_exp(p[1])
                for h in np.arange(len(p)-2):
                    for q in np.arange(power_t[h]+1): ecj.convolve_exp(p[h])
        else:
            if power_t is None:
                ecj = pp.ExponentialConvolution(p(branch=0)[1:],p(branch=0)[0])
                for b in np.arange(len(p.first_index["branch"])-1)[1:]:
                    branch_par = p(branch=b)
                    ecj.branch_path(branch_par[1],branch_par[0])
                    for h in np.arange(len(branch_par))[2:]:
                        ecj.convolve_exp(branch_par[h],branch=b)
            else:
                p_b0 = p(branch=0)
                pt_b0 = power_t(branch=0) #power_t must also be an irrarray
                
                ecj = pp.ExponentialConvolution([p_b0[1]],p_b0[0])
                if pt_b0[1] > 0:
                    for q in np.arange(pt_b0[1]): ecj.convolve_exp(p_b0[1])
                for h in np.arange(len(p_b0))[2:]:
                    for q in np.arange(pt_b0[h]+1): ecj.convolve_exp(p[h])
                
                for b in np.arange(len(p.first_index["branch"])-1)[1:]:
                    branch_par = p(branch=b)
                    pt_bi = power_t(branch=b)
                    ecj.branch_path(branch_par[1],branch_par[0])
                    if pt_bi[1]>0: 
                        for q in np.arange(pt_bi[1]): ecj.convolve_exp(branch_par[1],branch=b)
                    for h in np.arange(len(branch_par))[2:]:
                        for q in np.arange(pt_bi[h]+1):
                            ecj.convolve_exp(branch_par[h],branch=b)
                
        y = ecj.eval(x)
        del ecj
        
        return y
    
    @classmethod
    def fit_eci(cls,x,y,n_min=3,n_max=5,method=None,routine="least_squares",g1=None,g2=None,rms_limits=[None,None],auto_stop=False,rms_tol=1e-2):
        
        params = []
        rms = []
        
        y_norm = np.nansum(y)
        if np.isnan(y_norm) or np.isinf(y_norm) or y_norm==0:
            return None, None
        else:
            yb = y/y_norm
        
        for i in np.arange(n_min,n_max+1):
            p0 = 0.2+np.arange(i)*0.03
            p0 = np.append(1.,p0)
            
            lower_bounds = [-np.inf]
            for q in np.arange(i): lower_bounds.append(0.0)
            upper_bounds = [np.inf for q in np.arange(i+1)]
            if g1 is not None:
                p0[1] = g1
                lower_bounds[1] = g1*0.9
                upper_bounds[1] = g1*1.1
            if g2 is not None:
                p0[2] = g2
                lower_bounds[2] = g2*0.9
                upper_bounds[2] = g2*1.1
            
            if routine == "minimize":
                error = lambda p,x,y: np.sum(np.power(cls.eci(x,p)-y,2))
                res = minimize(error,p0,args=(x,yb),method=method)
                p = res.x
            elif routine == "least_squares":
                residuals = lambda p,x,y: cls.eci(x,p) - y
                res = least_squares(residuals,p0,args=(x,yb),method=method,bounds=(lower_bounds,upper_bounds))
                p = res.x
            p[0] *= y_norm
            params.append(p)
            rms.append(np.sqrt(np.sum(np.power((cls.eci(x,p)-y)[rms_limits[0]:rms_limits[1]],2))))
            
            if auto_stop and i>n_min:
                delta_rms_rel = np.abs(rms[-1]-rms[-2])/rms[-2]
                if delta_rms_rel<rms_tol: break
        
        return params,rms
        
    @classmethod
    def fit_eci_branching(cls,x,y,stim,dt,
                          n_hops_min=3,n_hops_max=5,
                          n_branches_max=5,
                          rms_limits=[None,None],auto_stop=False,rms_tol=1e-2,
                          method=None,routine="least_squares"):
        rms = []
        
        y_norm = np.sum(y)
        if np.isnan(y_norm) or np.isinf(y_norm) or y_norm==0:
            return None, None, None
        else:
            yb = y/y_norm
        #yb = y/y_norm
        stim_norm = np.sum(stim)
        stimb = stim/stim_norm
        
        p0_tot_prev_ = np.array([])
        p_prev = np.array([])
        n_in_prev = np.array([],dtype=int)
                
        for n_branches in np.arange(0,n_branches_max):
            
            rms_cur_b = []
            for i in np.arange(n_hops_min,n_hops_max+1):
                p0_cur_b_ = 0.2+np.arange(i)*0.03+n_branches*0.02
                if n_branches == 0:
                    A0 = 1.
                else:
                    A0 = (-1)**n_branches
                p0_cur_b_ = np.append(A0,p0_cur_b_)
                
                p0_tot_ = np.append(p0_tot_prev_,p0_cur_b_)
                p0_tot = mf.struct.irrarray(p0_tot_,[np.append(n_in_prev,i+1)],["branch"])
                
                lower_bounds = -np.inf*np.ones_like(p0_tot)
                upper_bounds = np.inf*np.ones_like(lower_bounds)
                special_idx = np.append(0,np.cumsum(n_in_prev))
                for q in np.arange(len(lower_bounds)):
                    if q not in special_idx:
                        lower_bounds[q] = 0.0
                    else:
                        branch_idx = np.where(special_idx==q)[0][0]
                        #if branch_idx%2==1: upper_bounds[q]=0.0
                        #elif branch_idx%2==0 and n_branches>0: lower_bounds[q]=0.0
                        
                if routine == "minimize":
                    error = lambda p,x,y: np.sum(np.power(pp.convolution(stimb,cls.eci(x,p),dt,8)-y,2))
                    res = minimize(error,p0_tot,args=(x,yb),method=method)
                    p_cur_b = res.x
                elif routine == "least_squares":
                    residuals = lambda p,x,y: pp.convolution(stimb,cls.eci(x,p),dt,8)-y
                    #residuals = cls.residuals_branching_least_squares
                    res = least_squares(residuals,p0_tot,args=(x,yb),method=method,bounds=(lower_bounds,upper_bounds))
                    p_cur_b = res.x
                rms_cur_b.append(np.sqrt(np.sum(np.power((cls.eci(x,p_cur_b)-yb)[rms_limits[0]:rms_limits[1]],2))))
                
                if auto_stop and i>n_hops_min:
                    delta_rms_rel = np.abs(rms_cur_b[-1]-rms_cur_b[-2])/rms_cur_b[-2]
                    if delta_rms_rel<rms_tol:break

            p0_tot_prev_= p0_tot_
            rms.append(rms_cur_b[-1])
            
            if auto_stop and n_branches>0:
                delta_rms_rel = (rms[-1]-rms[-2])/rms[-2]
                if np.abs(delta_rms_rel)<rms_tol and delta_rms_rel<0:
                    n_in_prev = np.append(n_in_prev,i+1)
                    p_prev = p_cur_b
                    break
                elif np.abs(delta_rms_rel)<rms_tol and delta_rms_rel>=0: 
                    rms.pop(-1)
                    break
                else:
                    n_in_prev = np.append(n_in_prev,i+1)
                    p_prev = p_cur_b
            else:
                n_in_prev = np.append(n_in_prev,i+1)
                p_prev = p_cur_b
        
        special_idx = np.append(0,np.cumsum(n_in_prev))
        for j in special_idx[:-1]:
            #FIXME BE CAREFUL WITH THIS. IF YOU HAVE REJOINING OF BRANCHES, IT
            # IS NOT GOING TO BEHAVE WELL!
            p_prev[j] *= y_norm/stim_norm
            
        gc.collect()
            
        return p_prev,n_in_prev,rms
    
    @classmethod    
    def residuals_branching_least_squares(cls,p,x,y,stimb,dt): 
        r = pp.convolution(stimb,cls.eci(x,p),dt,8)-y
        #r = np.convolve(stimb,cls.eci(x,p),mode="same")*dt-y
        return r
        
    def cluster_and_refit(self,p,n_branch_p,epsilon=1e-3):
        # Scan and find clusterable gammas within each branch
        #just do a np.around
                        
                
                
        # Write constraints to keep those gammas together
        
        # Do a least squares minimization
        return None
     
    @staticmethod   
    def error(p,x,y,f):
        e = np.sum(np.power(y-f(x,p),2))
        for a in p[1:]:
            e+=1e6*(np.sign(a-100.)+1.)
        
        return e        
        
    @classmethod
    def fit_many_single_stim(cls,xs,ys,p0=None,method=None,g1=None,g2=None):
        if p0 is None:
            p0 = [0.3,0.4]
            lower_bounds = [0.,0.]
            upper_bounds = [np.inf,np.inf]
            if g1 is not None: 
                p0[0] = g1
                # g1 not None means I want to constraint g1 to be that value
                lower_bounds[0] = g1
                upper_bounds[0] = g1+g1/10
            if g2 is not None: 
                p0[1] = g2
                lower_bounds[1] = g2
                upper_bounds[1] = g2+g2/10
            
            n_y = len(ys)
            for i in np.arange(n_y): 
                p0.append(1.0)
                lower_bounds.append(-np.inf)
                upper_bounds.append(np.inf)
                p0.append(0.01)
                lower_bounds.append(0)
                upper_bounds.append(np.inf)
        
        #res = minimize(cls.error_multi,p0,args=(xs,ys,cls._exp_conv_3b),method=method) #TODO CURRENTLY ONLY WITH ec3b
        res = least_squares(cls.ec3b_residuals_multi,p0,args=(xs,ys),method=method,bounds=(lower_bounds,upper_bounds))
        p = res.x
        
        return p
    
    @classmethod    
    def ec3b_residuals_multi(cls,p,xs,ys):
        '''For fits with least_squares, multiple ec3b at a time.'''
        n_pts = 0
        n_cumsum = []
        for x in xs:
            n_cumsum.append(n_pts)
            n_pts+=len(x)
        n_cumsum.append(n_pts)        
        out = np.zeros(n_pts)
        
        g1 = p[0]
        g2 = p[1]
        
        for i in np.arange(len(xs)):
            A = p[2+2*i]
            g3 = p[3+2*i]
            g13 = g1-g3
            g23 = g2-g3
            out[n_cumsum[i]:n_cumsum[i+1]] = cls.ec3b(xs[i],[A,g3,g13,g23]) - ys[i]
            
        return out
        
    @classmethod
    def ec4b_residuals(cls,):
        return None
   
    @staticmethod
    def error_multi(p,xs,ys,f):
        g1 = p[0]
        g2 = p[1]
        
        n_y = len(ys)
        e = 0.0
        for i in np.arange(n_y):
            A = p[2+2*i]
            g3 = p[3+2*i]
            g13 = g1-g3
            g23 = g2-g3
            e += np.sum(np.power(f(xs[i],[A,g3,g13,g23])-ys[i],2))
        
        return e
        
    @staticmethod    
    def _exp_conv_2_sqstim(t,p):
        A = p[0]
        tau1 = p[1]
        tau2 = p[2]
        delta = p[3]
        
        s = np.where(t<delta,t,delta)
        y1 = tau1 * np.exp(-t/tau1) * (np.exp(s/tau1)-1.0)        
        y2 = tau2 * np.exp(-t/tau2) * (np.exp(s/tau2)-1.0)
        y =   A * tau1*tau2/(tau1-tau2) * (y1-y2)      

        
        return y   
        
    @staticmethod
    def contiguously_satisfied(condition,n):
        if np.any(condition):
            contcond = np.diff(np.where(np.concatenate(([condition[0]],
                                        condition[:-1] != condition[1:],
                                        [True])))[0])[::2]
            return np.max(contcond)>n
        else:
            return False
    
    @classmethod
    def nan_ok(cls,a,n):
        if len(a.shape)==1:
            return not cls.contiguously_satisfied(a>0.5,n)
        elif len(a.shape)==2:
            ok = np.zeros(a.shape[1])
            for i in np.arange(a.shape[1]):
                ok[i] = not cls.contiguously_satisfied(a[:,i]>0.5,n)
            return ok
        else:
            raise ValueError("nan_ok max dimensionality is 2")
            
