import re, numpy as np, json, os, pickle, matplotlib.pyplot as plt
import pumpprobe as pp, wormbrain as wormb, wormdatamodel as wormdm
#For qvalues
from multipy.fdr import qvalue as fdrqvalue
from scipy.stats import binom, kstest, combine_pvalues
from statsmodels.stats.weightstats import ttost_ind as ttost_ind
from scipy.signal import savgol_coeffs

class Funatlas:

    nan_th = 0.05
    '''Global nan threshold'''
    
    fname_neuron_ids = "aconnectome_ids.txt"
    '''Name of file containing full list of neurons'''
    fname_ganglia = "aconnectome_ids_ganglia.json"
    '''Name of the file containing the list of neurons divided in ganglia'''
    fname_senseoryintermotor = "sensoryintermotor_ids.json"
    '''Name of the file containinf the list of neurons divided into sensory motor and interneurons'''
    fname_innexins = "GenesExpressing-unc-7-unc-9-inx-_-eat-5-thrs2.csv"
    '''Name of the file containing the expression levels of innexins'''
    fname_neuropeptides = "GenesExpressing-neuropeptides.csv"
    '''Name of the file containing the expression levels of neuropeptides'''
    fname_neuropeptide_receptors = "GenesExpressing-neuropeptide-receptors.csv"
    '''Name of the file containing the expression levels of neuropeptide receptors'''
    module_folder = "/".join(pp.__file__.split("/")[:-1])+"/"
    '''Folder of the pumpprobe module'''
    
    ds_list_used_fname = "funatlas_list_used.txt"
    ds_tags_lists = None
    
    aconn_sources = [#{"type": "white", "fname": "aconnectome.json", "ids_fname":"aconnectome_ids.txt"},
                     {"type": "whiteA", "fname": "aconnectome_white_1986_whole.csv"},
                     {"type": "whiteL4", "fname": "aconnectome_white_1986_L4.csv"},
                     {"type": "witvliet", "fname": "aconnectome_witvliet_2020_7.csv"},
                     {"type": "witvliet", "fname": "aconnectome_witvliet_2020_8.csv"}
                     ]
                     
    esconn_sources = [{"type": "bentley", "fname": "esconnectome_monoamines_Bentley_2016.csv", "transmitter_type": "monoamines"},
                      {"type": "bentley", "fname": "esconnectome_neuropeptides_Bentley_2016.csv", "transmitter_type": "neuropeptides"},
                     ]
                     
    def __init__(self,merge_bilateral=False,merge_dorsoventral=False,
                 merge_numbered=False,merge_AWC=False,verbose=True,
                 *args,**kwargs):
        '''Class initialization.
        
        Parameters
        ----------
        merge_bilateral: bool (optional)
            Whether to merge bilateral pairs. Default: False.
        merge_dorsoventral: bool (optional)
            Whether to merge dorsoventral pairs. Default: False.
        merge_numbered: bool (optional)
            Whether to merge numbered neuron sets. Default: False.
        '''         
        
        # Load the full list of neuronal ids from file       
        self._neuron_ids = np.loadtxt(self.module_folder+self.fname_neuron_ids,
                                     dtype=str)[:,1]
        
        # Set class options
        self.merge_bilateral = merge_bilateral  
        self.merge_dorsoventral = merge_dorsoventral
        self.merge_numbered = merge_numbered
        self.merge_AWC = merge_AWC
        # Reminder of the convention for AWCON and AWCOFF, since the anatomical
        # data has AWCL and AWCR.
        if not merge_AWC and not merge_bilateral:
            print("Using AWCOFF->AWCL and AWCON->AWCR.")
        if merge_numbered and verbose:
            print("Funatlas: Note that IL1 and IL2 will not be merged, as "+\
                   "well as VB1, VA1, VA11 because "+\
                   "they are different from the other VB and VA.")
        self.verbose = verbose
        
        # Compute the approximated (reduced) list of neuronal ids, e.g.
        # orphaning all the labels of their L/Rs.
        self.neuron_ids = self.approximate_ids(
                                self._neuron_ids,self.merge_bilateral,
                                self.merge_dorsoventral,self.merge_numbered,
                                self.merge_AWC)
        self.neuron_ids = np.unique(self.neuron_ids)
        self.n_neurons = len(self.neuron_ids)
        
        self.ganglia, self.head_ids, self.pharynx_ids = self.load_ganglia()
        self.sim, self.SIM_head_ids = self.load_sim_head()
        self.head_ids = self.approximate_ids(
                                self.head_ids,self.merge_bilateral,
                                self.merge_dorsoventral,self.merge_numbered,
                                self.merge_AWC)
        self.SIM_head_ids = self.approximate_ids(
                                self.SIM_head_ids,self.merge_bilateral,
                                self.merge_dorsoventral,self.merge_numbered,
                                self.merge_AWC)
        self.head_ids = np.unique(self.head_ids)
        self.SIM_head_ids = np.array(self.SIM_head_ids)
        self.head_ai = self.ids_to_i(self.head_ids)
        self.SIM_head_ai = self.ids_to_i(self.SIM_head_ids)
        
        self.pharynx_ids = self.approximate_ids(
                                self.pharynx_ids,self.merge_bilateral,
                                self.merge_dorsoventral,self.merge_numbered,
                                self.merge_AWC)
        self.pharynx_ids = np.unique(self.pharynx_ids)
        self.pharynx_ai = self.ids_to_i(self.pharynx_ids)
        
        
        # Load the anatomical connectome
        self.load_aconnectome_from_file()
        
        # Load the extrasynaptic connectome
        self.load_extrasynaptic_connectome_from_file()
        
        # Load innexin expression levels from file.
        self.load_innexin_expression_from_file()
        
        self.ds_list = None
        '''Source datasets for this atlas'''
        
        self.atlas_i = None
        '''atlas-indices of the neurons in each dataset'''
        
        self.labels = []
        '''Original labels'''
        
        self.labels_confidences = []
        '''Confidences of neuron labelings'''
        
        self.fconn = []
        '''list of Fconn objects of all the datasets'''
        
        self.stim_neurons_ai = []
        '''list of the atlas-indices of the stimulated neurons'''
        
        self.sig = []
        '''list of Signal objects of all the datasets'''
        
        self.brains = []
        '''list of Brains objects of all the datasets'''
        
        self.raw_resp = []
        for k in np.arange(self.n_neurons):
            self.raw_resp.append([[] for j in np.arange(self.n_neurons)])
        
    @classmethod
    def from_datasets(
            cls,ds_list,
            merge_bilateral=None,merge_dorsoventral=None,merge_numbered=None,
            merge_AWC=None,
            signal="green",signal_kwargs={},load_signal=True,verbose=True,
            ds_tags=None,ds_exclude_tags=None,
            enforce_stim_crosscheck=False):
        '''Create and populate an instance of Funatlas from the source 
        datasets.
        
        Parameters
        ----------
        ds_list: str or list of str
            Either name of the file containing the list of datasets, or 
            directly a list of dataset folders.
        merge_bilateral: bool (optional)
            Whether to merge bilateral pairs. Default: None, to keep the 
            __init__'s default.
        merge_dorsoventral: bool (optional)
            Whether to merge dorsoventral pairs. Default: None.
        merge_numbered: bool (optional)
            Whether to merge numbered neuron sets. Default: None.
        signal: str (optional)
            Which type of signal to use. Possible values: green, ratio. 
            Default: green.
        signal_kwargs: dict (optional)
            kwargs to be passed to the signal classes. Default: {}.
        load_signal: bool (optional)
            Whether to load the signal objects. Default: False.
        ds_tags: str (optional)
            Space-separated tags to select datasets. Default: None.
        enforce_stim_crosscheck: bool (optional)
            If True, stimulations that do not pass the artifact cross-checked
            are not considered (Fconn.stim_neurons[ie] is set to -2).
            Default: False.
            
        Returns
        -------
        funatlas: Funatlas
            Instance of the Funatlas class.
        
        '''
        # Create an instance of the Funatlas
        funatlas = cls(merge_bilateral,merge_dorsoventral,
                       merge_numbered,merge_AWC)
        
        if type(ds_list) == str:
            # Interpret it as the filename containing the list of datasets
            funatlas.ds_list, funatlas.ds_tags_lists =\
             cls.load_ds_list(ds_list,ds_tags,ds_exclude_tags,return_tags=True)
            funatlas.ds_tags = ds_tags
            funatlas.ds_exclude_tags = ds_exclude_tags
        else:
            # Interpret it as a list of dataset folders
            assert type(ds_list) == list
            funatlas.ds_list = ds_list
            
        n_ds = len(funatlas.ds_list)
        if n_ds>0: 
            funatlas.atlas_i = []
        else: print("No datasets listed."); quit()
        
        # Iterate over the datasets and load several things.
        disc_crosscheck = np.zeros(n_ds)
        for i_ds in np.arange(n_ds):
            folder = funatlas.ds_list[i_ds]
            
            ## Load this dataset's data
            # Get the Fconn objects
            fconn = pp.Fconn.from_file(folder,verbose=verbose)
            if enforce_stim_crosscheck:
                # If required, discard the stimulations that don't pass the 
                # artifact cross-check
                excl = 0
                excl_autoresp = 0
                valid_stim = 0
                autoresp = np.zeros(fconn.n_stim,dtype=bool)
                for _ie in np.arange(fconn.n_stim):
                    autoresp[_ie] = fconn.stim_neurons[_ie] in fconn.resp_neurons_by_stim[_ie]
                    '''if fconn.stim_neurons[_ie]!=-2: 
                        valid_stim += 1
                        if not (fconn.targeted_neuron_hit[_ie]):# or autoresp[_ie]):
                            excl += 1
                        if not fconn.targeted_neuron_hit[_ie] and autoresp[_ie]:
                            excl_autoresp += 1
                print("Discarding",np.around(excl/valid_stim,2),"out of valid stim",np.around(valid_stim/fconn.n_stim,2),"(",valid_stim,")")
                print(np.sum(autoresp),np.sum(np.logical_or(fconn.targeted_neuron_hit,autoresp)),valid_stim)
                #print("Discarding from non-2 targets",np.around(np.sum(np.logical_and(~fconn.targeted_neuron_hit,fconn.stim_neurons!=-2))/np.sum(fconn.stim_neurons!=-2),2))'''
                if valid_stim>0: 
                    disc_crosscheck[i_ds] = excl/valid_stim
                else:
                    disc_crosscheck[i_ds] = 0
                fconn.stim_neurons[np.logical_and(~fconn.targeted_neuron_hit,~autoresp)] = -2
            funatlas.fconn.append(fconn)
            
            # Get the labels and transform them in atlas indices
            cervello = wormb.Brains.from_file(folder,ref_only=True,
                                         verbose=False)
            funatlas.brains.append(cervello)
            ids,ids_confidences = cervello.get_labels(0,return_confidences=True)
            funatlas.labels.append(ids)
            funatlas.atlas_i.append(funatlas.ids_to_i(ids)) #Does't ids to i use atlas_i??
            funatlas.labels_confidences.append(ids_confidences)
            
            # As a utility, transform also the stimulated neuron indices to 
            # atlas indices.
            mask = fconn.stim_neurons>=0
            stim_neurons_ai = np.copy(fconn.stim_neurons)
            stim_neurons_ai[mask] = funatlas.atlas_i[-1][fconn.stim_neurons[mask]]
            # Include also the complement labels from the manually identified
            # targets, for the -3 targeted neurons that could be identified 
            # but were not present in the reference volume.
            if np.any(fconn.stim_neurons==-3):
                compl_labs = np.array(fconn.stim_neurons_compl_labels)[fconn.stim_neurons==-3]
                compl_ai = funatlas.ids_to_i(compl_labs)
                stim_neurons_ai[fconn.stim_neurons==-3] = compl_ai
            
            funatlas.stim_neurons_ai.append(stim_neurons_ai)
                        
            # Get the Signal objects
            if load_signal:
                if signal == "ratio":
                    sig = wormdm.signal.Signal.from_signal_and_reference(
                                           folder,"green","red",**signal_kwargs)
                else:
                    sig = wormdm.signal.Signal.from_file(
                                           folder,signal,**signal_kwargs)
                funatlas.sig.append(sig)
        
        if enforce_stim_crosscheck: print("average disc",np.average(disc_crosscheck))
        return funatlas
        
    def export_to_txt(self,folder=None):
        '''Export to text files.
        '''
        
        if folder is None:
            print("Funatlas.export_to_txt(): specify folder. Done nothing.")
            return None
        
        n_ds = len(self.ds_list)
        for i_ds in np.arange(n_ds):
            ds = self.ds_list[i_ds]
            
            y = self.sig[i_ds].data
            t = np.arange(len(y))*self.fconn[i_ds].Dt
            
            lbl = self.labels[i_ds]
            
            stim_vol_i = self.fconn[i_ds].i0s + self.fconn[i_ds].shift_vols
            
            stim_neu = self.fconn[i_ds].stim_neurons
            
            np.savetxt(folder+str(i_ds)+"_gcamp.txt",y)
            np.savetxt(folder+str(i_ds)+"_t.txt",t)
            f = open(folder+str(i_ds)+"_labels.txt","w")
            for l in lbl: f.write(l+"\n")
            f.close()
            np.savetxt(folder+str(i_ds)+"_stim_volume_i.txt",stim_vol_i,fmt="%d")
            np.savetxt(folder+str(i_ds)+"_stim_neurons.txt",stim_neu,fmt="%d")
            f = open(folder+str(i_ds)+"_ds_name.txt","w")
            f.write(ds)
            f.close()
        
        
    def set_raw_response(self,ds,to_i,from_i,response):
        '''This will likely not be used. Working directly with the Signal
        objects.'''
        self.raw_resp[to_i][from_i].append({"ds":ds,"data":response})
        
    def get_raw_responses(self,to_i,from_i,data_only=True):
        '''This will likely not be used. Working directly with the Signal
        objects.'''
        if data_only:
            return [r["data"] for r in self.raw_resp[to_i][from_i]]
        else:
            return self.raw_resp[to_i][from_i]
                                     
    def approximate_ids(self,ids,merge_bilateral=True,merge_dorsoventral=True,
                 merge_numbered=True,merge_AWC=False):
        '''Approximate the neural IDs, by orphaning them of the bilateral L/R
        and/or the dorso-ventral V/D and/or of the numbering for the 
        undistinguishable neurons in the retrovesicular ganglion and ventral
        nerve cord.
        
        Parameters
        ----------
        ids: str or array_like of str
            IDs to be approximated.
        merge_bilateral: bool (optional)
            Whether to merge left/right pairs. Default: True.
        merge_dorsoventral: bool (optional)
            Whether to merge dorsal/ventral pairs, triplets, and quadruples. 
            Default: True.
        merge_numbered: bool (optional)
            Whether to merge the undistinguishable merged neurons in the 
            retrovesicular ganglion and ventral nerve cord. Note that VB1, VA1, 
            VA11 will not be merged because they are different from the other 
            VB and VA. IL1 and IL2 will also not be merged. Default: True.
            
        Returns
        -------
        out_ids: str or list of str
            Approximated IDs. Single string if ids was a string, list of strings
            if ids was an array_like of strings.
        '''
        
        # If ids is a string, convert it to a list of string for processing.
        if type(ids)!=str:
            iter_input=True
        else: 
            ids = [ids] 
            iter_input=False
            
        out_ids = []
        for in_id in ids:
            if in_id is None:
                out_ids.append("")
                continue
            out_id = in_id
                    
            if len(in_id)>0:
                if in_id in ["AWCOFF","AWCON"]:
                    # Separate treatment: The anatomical data calls them AWCL 
                    # and AWCR, not ON and OFF. Special treatment also in
                    # self.ids_to_i().
                    if merge_bilateral or merge_AWC:
                        out_id = "AWC"
                    elif not merge_AWC and not merge_bilateral:
                        if in_id=="AWCOFF": out_id="AWCL"
                        elif in_id=="AWCON": out_id="AWCR"
                        
                if merge_bilateral and in_id not in ["AQR"]:
                    if in_id[-1]=="L":
                        if in_id[:-1]+"R" in self._neuron_ids or\
                         in_id[:-2]+"D"+"R" in self._neuron_ids or\
                         in_id[:-2]+"V"+"R" in self._neuron_ids or\
                         in_id[:-2]+"D" in self._neuron_ids: 
                            out_id = in_id[:-1]+"_"
                    elif in_id[-1]=="R":
                        if in_id[:-1]+"L" in self._neuron_ids or\
                         in_id[:-2]+"D"+"L" in self._neuron_ids or\
                         in_id[:-2]+"V"+"L" in self._neuron_ids or\
                         in_id[:-2]+"D" in self._neuron_ids:
                            out_id = in_id[:-1]+"_"
                            
                if merge_dorsoventral:
                    if len(out_id)==4:
                        if out_id[-1]=="D":
                            # To make SMBD -> SMB_ (because there is SMBVL)
                            
                            if out_id[:-1]+"V" in self._neuron_ids or\
                             out_id[:-1]+"VL" in self._neuron_ids or\
                             out_id[:-1]+"VR" in self._neuron_ids:
                                out_id = out_id[:-1]+"_"
                                
                        elif out_id[-1]=="V":
                            if out_id[:-1]+"D" in self._neuron_ids or\
                             out_id[:-1]+"DL" in self._neuron_ids or\
                             out_id[:-1]+"DR" in self._neuron_ids:
                                out_id = out_id[:-1]+"_"
                    
                    if len(out_id)==5:
                        if out_id[-2]=="V":
                            # To make OLQVL -> OLQ_L
                            if out_id[:-2]+"D"+out_id[-1] in self._neuron_ids or\
                              out_id[:-2]+"D"+"L" in self._neuron_ids or\
                              out_id[:-2]+"D"+"R" in self._neuron_ids or\
                              out_id[:-2]+"D" in self._neuron_ids:
                                out_id = out_id[:-2]+"_"+out_id[-1]
                                
                        elif out_id[-2]=="D":
                            # To make OLQVL -> OLQ_L
                            if out_id[:-2]+"V"+out_id[-1] in self._neuron_ids or\
                             out_id[:-2]+"V"+"L" in self._neuron_ids or\
                             out_id[:-2]+"V"+"R" in self._neuron_ids or\
                             out_id[:-2]+"V" in self._neuron_ids:
                                out_id = out_id[:-2]+"_"+out_id[-1]
                                
                            
                if merge_numbered:
                    if len(out_id)>2 and\
                      out_id not in ["VB1","VA1","VA11","IL1","IL2"]:
                        if out_id[-2:].isdigit():
                            out_id = out_id[:-2]+"_"
                        elif out_id[-1].isdigit():
                            out_id = out_id[:-1]+"_"
                            
                    
            out_ids.append(out_id)
            
        if iter_input: return out_ids
        else: return out_ids[0]
        
    def ids_to_i(self,ids):
        '''Converts IDs to atlas indices.
        
        Parameters
        ----------
        ids: string or array_like of strings 
            IDs of the neurons.
        
        Returns
        -------
        atlas_i: int or numpy.ndarray of int
            Atlas-indices of the input IDs.
        '''
        
        if type(ids)==str: ids = [ids]; was_string = True
        else: was_string = False
        
        n_ids = len(ids)
        
        atlas_i = -1*np.ones(n_ids,dtype=int)
        # Approximate the input identities according to the same settings as for
        # this atlas' instance.
        approx_ids = self.approximate_ids(
                            ids,self.merge_bilateral,self.merge_dorsoventral,
                            self.merge_numbered,self.merge_AWC)

        atlas_ids_B = np.copy(self.neuron_ids)
        for i in np.arange(len(atlas_ids_B)):
            atlas_ids_B[i] = atlas_ids_B[i].replace("_","")
            
        for i in np.arange(n_ids):
            approx_ids[i] = approx_ids[i].replace("_","")
                
        for i in np.arange(n_ids):
            matched = np.where(atlas_ids_B==approx_ids[i])[0]
            if matched.shape[0]>0:
                atlas_i[i] = matched[0]
        
        if was_string:
            return atlas_i[0]
        else:
            return atlas_i
            
    def i_to_ai(self,i,i_ds):
        '''Converts dataset-specific indices to atlas indices (ai).
        
        Parameters
        ----------
        i: int, array_like of int
            Dataset-specific indices.
        i_ds: int
            Dataset.
            
        Returns
        -------
        ai: int or numpy.ndarray
            Atlas indices corresponding to the input dataset-specific indices.
            If i was a scalar, ai is a scalar. Otherwise, ai is a numpy.ndarray.
        '''
        
        try: len(i); was_scalar = False
        except: i = [i]; was_scalar = True
        i = np.array(i)
        
        mask = i>=0
        ai = np.copy(i)
        ai[mask] = self.atlas_i[i_ds][i[mask]]
        
        if was_scalar: return ai[0]
        else: return ai        
        
    @staticmethod
    def load_ds_list(fname,tags=None,exclude_tags=None,return_tags=False):
        '''Loads the list of dataset folder names given the filename of a 
        text file containing a folder name per line. Comments start with # (both
        for whole line and for annotations after the folder name).
        
        Parameters
        ----------
        fname: str
            Name of the txt file containing the list of datasets.
        tags: str (optional)
            Space-separated tags to select datasets. Default: None.
        exclude_tags: str (optional)
            Space-separated tags to exclude in the dataset selection. Default:
            None.
        
        Returns
        -------
        ds_list: list of str
            List of the dataset folder names.
        '''
        
        f = open(fname,"r")
        ds_list = []
        ds_tags_lists = []
        if tags is not None: tags = tags.split(" ")
        if exclude_tags is not None: exclude_tags = exclude_tags.split(" ")
        
        for l in f.readlines():
            if l[0] not in ["#","\n"]: # Ignore commented lines
                # Remove commented annotations
                l2 = l.split("#")[0]
                # Get tags
                if len(l.split("#"))==0: 
                    # There are no tags
                    continue
                tgs = l.split("#")[1].split(" ")
                for it in np.arange(len(tgs)):
                    tgs[it] = re.sub(" ","",tgs[it])
                    tgs[it] = re.sub("\n","",tgs[it])
                    tgs[it] = re.sub("\r","",tgs[it])
                    tgs[it] = re.sub("\t","",tgs[it])
                if tags is not None:
                    ok = [t in tgs or t=="" for t in tags]
                    if not np.all(ok): continue
                if exclude_tags is not None:
                    not_ok = [t in tgs and t!="" for t in exclude_tags]
                    if np.any(not_ok): continue
                
                # Remove blank spaces, newlines, and tabs
                l2 = re.sub(" ","",l2)
                l2 = re.sub("\n","",l2)
                l2 = re.sub("\r","",l2)
                l2 = re.sub("\t","",l2)
                
                # Complete folder path with last / if necessary
                if l2[-1]!="/":l2+="/" 
                
                ds_tags_lists.append(tgs)
                ds_list.append(l2)
        f.close()
        
        if return_tags:
            return ds_list, ds_tags_lists
        else:
            return ds_list
        
    def load_ganglia(self):
        # Load the whole object
        g_f = open(self.module_folder+self.fname_ganglia,"r")
        ganglia = json.load(g_f)
        g_f.close()
        
        # Remove M2, as it is absent from the other anatomical datasets
        #ind = ganglia["posterior pharyngeal bulb"].index("M2")
        #ganglia["posterior pharyngeal bulb"].pop(ind)
        #if self.verbose:
        #    print("Funatlas: Removing M2 from ganglia information.")
        
        # Make a flattened list of neurons that are in the head
        head_ids = []
        for k in ganglia["head"]:
            for neu_id in ganglia[k]:
                head_ids.append(neu_id)
                
        pharynx_ids = []
        for k in ganglia["pharynx"]:
            for neu_id in ganglia[k]:
                pharynx_ids.append(neu_id)
        
        return ganglia, head_ids, pharynx_ids
    
    def load_sim_head(self):
    #load the ganglia and sim object
    	g_f = open(self.module_folder+self.fname_ganglia,"r")
    	ganglia = json.load(g_f)
    	g_f.close()
    	sim_f = open(self.module_folder+self.fname_senseoryintermotor,"r")
    	sim = json.load(sim_f)
    	sim_f.close
    	
    	head_ids = []
    	for k in ganglia["head"]:
    	    for neu_id in ganglia[k]:
    	        head_ids.append(neu_id)
    	SIM_head_ids = []
    	categories = ["Sensory", "Inter", "Motor"]
    	for c in categories:
    	    for neu_id_sim in sim[c]:
    	        if neu_id_sim in head_ids:
    	            SIM_head_ids.append(neu_id_sim)
    	
    	return sim, SIM_head_ids
        
    
    def reduce_to_head(self,A):
        '''Returns the submatrix or subarray of A corresponding to the head 
        neurons only.
        
        Parameters
        ----------
        A: numpy.ndarray
            Matrix or array to be cut.
        
        Returns
        -------
        A_new: numpy.ndarray
            Matrix reduced to the head indices.
        '''
        
        if len(A.shape)>=2:
            A_new = A[self.head_ai][:,self.head_ai]
        elif len(A.shape)==1:
            A_new = A[self.head_ai]
        return A_new 
    
    def reduce_to_SIM_head(self,A):
        '''Returns the submatrix or subarray of A corresponding to the head 
        neurons only sorted by the sensory, inter and motor neurons
        
        Parameters
        ----------
        A: numpy.ndarray
            Matrix or array to be cut.
        
        Returns
        -------
        A_new: numpy.ndarray
            Matrix reduced to the head indices.
        '''
        
        if len(A.shape)==2:
            A_new = A[self.SIM_head_ai][:,self.SIM_head_ai]
        elif len(A.shape)==1:
            A_new = A[self.SIM_head_ai]
        return A_new 
        
    def reduce_to_pharynx(self,A):
        '''Returns the submatrix or subarray of A corresponding to the pharynx 
        neurons only.
        
        Parameters
        ----------
        A: numpy.ndarray
            Matrix or array to be cut.
        
        Returns
        -------
        A_new: numpy.ndarray
            Matrix reduced to the pharynx indices.
        '''
        
        if len(A.shape)==2:
            A_new = A[self.pharynx_ai][:,self.pharynx_ai]
        elif len(A.shape)==1:
            A_new = A[self.pharynx_ai]
        return A_new 
        
    def ai_to_head(self,ais):
        '''Translate whole-body atlas-indices to positions of those indices
        in the head-restricted reference frame.
        
        Parameters
        ----------
        ais: array_like of int
            Whole-body atlas-indices.
        
        Returns
        -------
        head_ais: numpy.ndarray of int
            Translated indices.        
        '''
        
        ais_head = np.zeros_like(ais)
        for i in np.arange(len(ais)):
            ai_head = np.where(self.head_ai==ais[i])[0]
            if len(ai_head)>0:
                ais_head[i] = ai_head[0]
            else:
                ais_head[i] = -1
                
        return ais_head
        
        
    def get_occurrence_matrix(self, req_auto_response=False, stim_shift=0,
                              inclall=False):
        '''Returns two matrices that describe the multiple occurrences of the 
        stimulus/response of the same pair of neurons. The matrix occ1 counts 
        how many occurrences are present in the datasets, while the matrix occ2
        contains information about how to retrieve those occurrences from the
        Signal objects.
        
        Parameters
        ----------
        req_auto_response: bool (optional)
            Whether to only include responses of downstream neurons when the 
            targeted neuron itself also shows a recorded response. 
            Default: False.
        stim_shift: int (optional)
            Response index after stimulation. If stim_shift==1, then the 
            function looks for responses of neuron i to the next stimulation
            after the stimulation of neuron j. Looks for sensitization.
            Default: 0.
        inclall: bool (optional)
            If True, the matrices are going to include all the identified 
            neurons instead of only the responding neurons.
        
        Returns
        -------
        occ1: numpy.ndarray
            occ1[i,j] is the number of occurrences of the response of i 
            following a stimulation of j.
        occ2: numpy.ndarray of lists of dictionaries
            occ2[i,j] is a dictionary containing details to extract the 
            activities from the Signal objects. Keys: ds (dataset index), stim
            (stimulus index, ie), resp_neu_i (dataset-specific index of the
            responding neuron - not atlas-index).
        '''
        
        occ1 = np.zeros((self.n_neurons,self.n_neurons),dtype=int)
        occ2 = [[[] for j in np.arange(self.n_neurons)] for i in np.arange(self.n_neurons)]
        
        if stim_shift<0:
            raise ValueError("stim_shift can only be positive/causal.")
        
        # Iterate over datasets
        for i_ds in np.arange(len(self.ds_list)):
            # Iterate over the stimulations
            for ie in np.arange(len(self.stim_neurons_ai[i_ds])-stim_shift):
                # Get the atlas-index of the stimulated neuron, and don't do
                # anything if it wasn't identified.
                aj = self.stim_neurons_ai[i_ds][ie]
                if aj<0: continue
                
                if not inclall:
                    # Get the atlas-index of the responding neurons.
                    aii = self.i_to_ai(self.fconn[i_ds].resp_neurons_by_stim[ie+stim_shift],i_ds)
                else:
                    # Get the atlas-index of all the (identified) neurons in the
                    # dataset.
                    aii = self.i_to_ai(np.arange(self.fconn[i_ds].n_neurons),i_ds)
                    
                # Andy: if desired, require that the stim neuron responds
                # No, this is wrong, because it will be ok with ASHL responding
                # when ASHR is stimulated, if merge_bilateral==True.
                #if req_auto_response and (aj not in aii): continue
                if req_auto_response and \
                    self.fconn[i_ds].stim_neurons[ie] not in \
                        self.fconn[i_ds].resp_neurons_by_stim[ie]:
                        continue
                

                # Increase the counts in occ1.
                #occ1[aii[aii>=0],aj] += 1
                aii_u, aii_u_n = np.unique(aii[aii>=0],return_counts=True)
                occ1[aii_u,aj] += aii_u_n
                
                # Iterate over the responding neurons individually and populate
                # a list of dictionaries containing the index of the current
                # dataset, the index of the stimulation, and the 
                # dataset-specific index of the responding neuron.
                if not inclall:
                    neuron_list = self.fconn[i_ds].resp_neurons_by_stim[ie+stim_shift]
                else:
                    neuron_list = np.arange(self.fconn[i_ds].n_neurons)
                
                L = len(neuron_list)
                for l in np.arange(L):
                    i = neuron_list[l]
                    ai = aii[l]
                    if ai>=0:
                        occ2[ai][aj].append({"ds":i_ds,"stim":ie,"resp_neu_i":i})
                        if stim_shift>0:
                            occ2[ai][aj][-1]["shift_stim_aj"] = self.stim_neurons_ai[i_ds][ie+stim_shift]
            
        return occ1, np.array(occ2,dtype=object)
        
    def get_observation_matrix(self, req_auto_response=False):
        '''Returns a matrix that counts how many times the coupling between
        two neurons could have been observed, i.e. the times j was stimulated
        and i was being observed. Useful to normalize occ1 from 
        get_occurrence_matrix().
        
        Parameters
        ----------
        req_auto_response: bool (optional)
            Whether to only include cases in which the targeted neuron itself
            also shows a recorded response. 
            Default: False.
        
        Returns
        -------
        occ3: numpy.ndarray
            occ1[i,j] is the number of times the coupling from j to i could have
            been observed.
        '''
        
        occ3 = np.zeros((self.n_neurons,self.n_neurons),dtype=int)
        
        # Iterate over datasets
        for i_ds in np.arange(len(self.ds_list)):
            # Iterate over the stimulations
            
            # Get the number of times the connection of this pair could
            # actually be observed (number of times in which j was 
            # stimulated and i was observed).
            obs_ais = self.atlas_i[i_ds]
            obs_ais = obs_ais[obs_ais>=0]
            obs_ais_u, obs_ais_u_n = np.unique(obs_ais, return_counts=True)
            
            for ie in np.arange(len(self.stim_neurons_ai[i_ds])):
                # Get the atlas-index of the stimulated neuron, and don't do
                # anything if it wasn't identified.
                aj = self.stim_neurons_ai[i_ds][ie]
                if aj<0: continue
                
                if req_auto_response and \
                self.fconn[i_ds].stim_neurons[ie] not in \
                    self.fconn[i_ds].resp_neurons_by_stim[ie]:
                    continue

                occ3[obs_ais_u,aj] += obs_ais_u_n
                
        return occ3
        
    def get_observation_matrix_nanthresh(self, req_auto_response=False):
        occ3_nonan = np.zeros((self.n_neurons, self.n_neurons), dtype=int)
        '''Returns a matrix that counts how many times the coupling between
        two neurons could have been observed excluding traces that do not pass the nan threshold,
         i.e. the times j was stimulated
        and i was being observed. Useful to normalize occ1 from 
        get_occurrence_matrix().'''
        nan_th = self.nan_th
        print("CHECK NEW NAN THRESHOLD CHECK, CTRL+F nan_ok")

        # Iterate over datasets
        for i_ds in np.arange(len(self.ds_list)):
            # Iterate over the stimulations
            # Get the number of times the connection of this pair could
            # actually be observed (number of times in which j was
            # stimulated and i was observed).
            obs_ais_original = self.atlas_i[i_ds]
            neuron_list = np.arange(self.fconn[i_ds].n_neurons)

            for ie in np.arange(len(self.stim_neurons_ai[i_ds])):
                # Get the atlas-index of the stimulated neuron, and don't do
                # anything if it wasn't identified.
                aj = self.stim_neurons_ai[i_ds][ie]
                i0 = self.fconn[i_ds].i0s[ie]
                i1 = self.fconn[i_ds].i1s[ie]
                nan_mask = self.sig[i_ds].get_segment_nan_mask(i0, i1)
                #nan_selection = np.sum(nan_mask, axis=0) <= nan_th * (i1 - i0)
                nan_selection = pp.Fconn.nan_ok(nan_mask,nan_th * (i1 - i0))
                
                y = self.sig[i_ds].get_segment(i0,i1,baseline=False,normalize="none")
                presel = np.ones(len(nan_selection),dtype=bool)
                for i_ in np.arange(len(nan_selection)):
                    pre = self.sig[i_ds].get_loc_std(y[:,i_],8)
                    if pre==0.0:
                        presel[i_] = False
                
                obis_ais_above_nanth = obs_ais_original[np.where(np.logical_and(nan_selection,presel))]
                obs_ais = obis_ais_above_nanth[obis_ais_above_nanth >= 0]
                obs_ais_u, obs_ais_u_n = np.unique(obs_ais, return_counts=True)


                if aj < 0: continue

                if req_auto_response and \
                        self.fconn[i_ds].stim_neurons[ie] not in \
                        self.fconn[i_ds].resp_neurons_by_stim[ie]:
                    continue

                occ3_nonan[obs_ais_u, aj] += obs_ais_u_n

        return occ3_nonan

        
    def get_times_j_stimulated(self,aj=None,jid=None,req_auto_response=True):
        if jid is not None:
            aj = self.ids_to_i(jid)
        elif aj is None and jid is None:
            raise ValueError("You need to pass at least one out of aj and jid.")
        
        njstim = 0
        for ds in np.arange(len(self.ds_list)):
           
            if req_auto_response:
                 # Get all the ds_j corresponding to this aj 
                # (could be more than one!). Continue if there is none.
                js = np.where(self.atlas_i[ds]==aj)[0]
                if len(js)==0:continue
                jstim_ = 0
                for ie in np.arange(self.fconn[ds].n_stim):
                    if self.fconn[ds].stim_neurons[ie] in js:
                        if self.fconn[ds].stim_neurons[ie] in self.fconn[ds].resp_neurons_by_stim[ie]:
                            jstim_ += 1
                #jstim_ = len(np.where( self.stim_neurons_ai[ds] == aj *\
                #             [j in resp for resp in self.fconn[ds].resp_neurons_by_stim])[0])
            else:
                jstim_ = len(np.where( self.stim_neurons_ai[ds] == aj)[0])
            njstim += jstim_
            
        return njstim
        
    def get_times_all_j_stimulated(self,req_auto_response=True):
        
        njstim = np.zeros(self.n_neurons,dtype=int)
        for aj in np.arange(self.n_neurons):
            njstim[aj] = self.get_times_j_stimulated(
                            aj=aj,
                            req_auto_response=req_auto_response)
                            
        return njstim
        
    def get_times_i_observed(self,ai=None,iid=None,req_auto_response=False):
        if iid is not None:
            ai = self.ids_to_i(iid)
        elif ai is None and iid is None:
            raise ValueError("You need to pass at least one out of aj and jid.")
        
        niobs = 0
        for ds in np.arange(len(self.ds_list)):
            ok = not req_auto_response
            if req_auto_response:
                for ie in np.arange(len(self.fconn[ds].stim_neurons)):
                    stim_j = self.fconn[ds].stim_neurons[ie]
                    if stim_j in self.fconn[ds].resp_neurons_by_stim[ie]:
                        ok = True
            #if len(np.where(self.atlas_i[ds]==ai)[0])>0 and ok:
            #    niobs +=1 
            if ok: niobs += len(np.where(self.atlas_i[ds]==ai)[0])
        
        return niobs
        
    def get_times_all_i_observed(self,req_auto_response=False):
        niobs = np.zeros(self.n_neurons,dtype=int)
        for ai in np.arange(self.n_neurons):
            niobs[ai] = self.get_times_i_observed(ai,req_auto_response=req_auto_response)
            
        return niobs
        
    def get_bilateral_companions(self,neuron_ids=None):
        if neuron_ids is None:
            neuron_ids = self.neuron_ids
            
        if self.merge_bilateral or self.merge_dorsoventral or self.merge_numbered:
            print("Funatlas.get_bilateral_companions only without merging in Funatlas object.")
            return None
            
        companions = -1*np.ones(len(neuron_ids),dtype=int)
        for i in np.arange(len(neuron_ids)):
            nid = neuron_ids[i]
            nid_app = self.approximate_ids(
                            [nid],merge_bilateral=True,
                            merge_dorsoventral=False,merge_numbered=False)[0]
            for k in np.arange(len(self.neuron_ids)):
                nid2 = self.neuron_ids[k]
                if nid2 != nid:
                    nid2_app = self.approximate_ids(
                                [nid2],merge_bilateral=True,
                                merge_dorsoventral=False,merge_numbered=False)[0]
                    if nid2_app == nid_app:
                        companions[i] = k
                
        return companions
        
    def get_shuffling_sorter(self):
        rng = np.random.default_rng()
        shuffling_sorter = rng.permutation(self.n_neurons)
        return shuffling_sorter
    
    def shuffle_array(self, a, shuffling_sorter=None):
        if shuffling_sorter is None:
            shuffling_sorter = self.get_shuffling_sorter()
        
        if len(a.shape)==2:
            a = a[shuffling_sorter][:,shuffling_sorter]
        elif len(a.shape)==1:
            a = a[shuffling_sorter]
        else:
            raise ValueError("Implemented only for 1D and 2D arrays.")
            
        return a
        
    def cengen_ids_conversion(self,ids):
        
        if type(ids)==str:
            ids = [ids]
            was_scalar=True
        else:
            was_scalar=False
            
        
        names_out = []
        for iid in np.arange(len(ids)):
            if ids[iid]=="IL1":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["IL1_","IL1D_","IL1V_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["IL1"]
                else:
                    names = ["IL1R","IL1L","IL1DR","IL1DL","IL1VR","IL1VL"]
            elif ids[iid]=="IL2_DV":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["IL2D_","IL2V_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = []
                else:
                    names = ["IL2DR","IL2DL","IL2VR","IL2VL"]
            elif ids[iid]=="IL2_LR":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["IL2_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = []
                else:
                    names = ["IL2R","IL2L"]
            elif ids[iid]=="CEP":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["CEPD_","CEPV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["CEP__"]
                else:
                    names = ["CEPDR","CEPDL","CEPVR","CEPVL"]
            elif ids[iid]=="OLQ":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["OLQD_","OLQV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["OLQ__"]
                else:
                    names = ["OLQDR","OLQDL","OLQVR","OLQVL"]
            elif ids[iid]=="RMD_DV":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["RMDD_","RMDV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["RMD__"]
                else:
                    names = ["RMDDR","RMDDL","RMDVR","RMDVL"]
            elif ids[iid]=="RMD_LR":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["RMD_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["RMD_"]
                else:
                    names = ["RMDL","RMDR"]
            elif ids[iid]=="RME_DV":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["RMED","RMEV"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["RME_"]
                else:
                    names = ["RMED","RMEV"]
            elif ids[iid]=="RME_LR":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["RME_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["RME_"]
                else:
                    names = ["RMEL","RMER"]
            elif ids[iid]=="SMD":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["SMDD_","SMDV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["SMD__"]
                else:
                    names = ["SMDDR","SMDDL","SMDVR","SMDVL"]
            elif ids[iid]=="URY":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["URYD_","URYV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["URY__"]
                else:
                    names = ["URYDR","URYDL","URYVR","URYDL"]
            elif ids[iid]=="URA":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["URAD_","URAV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["URA__"]
                else:
                    names = ["URADR","URADL","URAVR","URADL"]
            elif ids[iid]=="SAA":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["SAAD_","SAAV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["SAA__"]
                else:
                    names = ["SAADR","SAADL","SAAVR","SAAVL"]
            elif ids[iid]=="SAB":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["SABD_","SABV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["SAB__"]
                else:
                    names = ["SABDR","SABDL","SABVR","SABVL"]
            elif ids[iid]=="SIA":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["SIAD_","SIAV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["SIA__"]
                else:
                    names = ["SIADR","SIADL","SIAVR","SIAVL"]
            elif ids[iid]=="SIB":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["SIBD_","SIBV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["SIB__"]
                else:
                    names = ["SIBDR","SIBDL","SIBVR","SIBVL"]
            elif ids[iid]=="SMB":
                if self.merge_bilateral and not self.merge_dorsoventral:
                    names = ["SMBD_","SMBV_"]
                elif self.merge_bilateral and self.merge_dorsoventral:
                    names = ["SMB__"]
                else:
                    names = ["SMBDR","SMBDL","SMBVR","SMBVL"]
            elif ids[iid]=="AWC_ON":
                if self.merge_AWC:
                    names = ["AWC"]
                else:
                    names = ["AWCON"]
            elif ids[iid]=="AWC_OFF":
                if self.merge_AWC:
                    names = ["AWC"]
                else:
                    names = ["AWCOFF"]
            else:
                names = [ids[iid]]
                
        names_out.append(names)
        
        if was_scalar:
            return names_out[0]
        else:
            return names_out
            
    def matrix_to_merged(self,A,**kwargs):
        n = A.shape[0]
    
        app_ids = np.unique(self.approximate_ids(self.neuron_ids,**kwargs))
        n2 = len(app_ids)
        A2 = np.zeros((n2,n2))
        count = np.zeros((n2,n2))
        
        for i in np.arange(n):
            for j in np.arange(n):
                iid = self.neuron_ids[i]
                jid = self.neuron_ids[j]
                iid2,jid2 = self.approximate_ids([iid,jid],**kwargs)
                i2 = np.where(iid2==app_ids)[0]
                j2 = np.where(jid2==app_ids)[0]
                if not np.isnan(A[i,j]):
                    A2[i2,j2] += A[i,j]
                    count[i2,j2] += 1
                
        return A2
            
    @staticmethod
    def get_nan_matrix_argsort(matrix,axis=-1):
        axis2 = 0 if axis==-1 else -1
        nansum = np.sum(np.isnan(matrix),axis=axis2)
        sorter = np.argsort(nansum)
        
        lim = np.where(nansum[sorter]==matrix.shape[axis])[0]
        if len(lim)>0:
            lim = lim[0]
        else:
            lim = matrix.shape[axis]
        
        return sorter,lim
        
    @classmethod
    def sort_matrix_nans(cls,matrix,axis=-1,return_all=False):
        sorter,lim = cls.get_nan_matrix_argsort(matrix,axis)
        
        matrix = matrix[sorter][:,sorter]
        
        if return_all:
            return matrix, sorter, lim
        else:
            return matrix
            
    @staticmethod
    def sort_matrix_pop_nans(matrix,return_all=False):
        sorter_j = np.where(~np.all(np.isnan(matrix),axis=0))[0]
        #sorter_i = np.where(~np.all(np.isnan(matrix),axis=1))[0]        
        sorter_i_actual = np.where(~np.all(np.isnan(matrix), axis=1))[0]
        lim = len(sorter_j) - 1
        sorter_i = np.copy(sorter_j)
        for i in np.arange(matrix.shape[0]):
            if i not in sorter_i and i in sorter_i_actual:
                sorter_i = np.append(sorter_i, i)

        #sorter_i = np.copy(sorter_j)
        #for i in np.arange(matrix.shape[0]):
        #    if i not in sorter_i:
        #        sorter_i = np.append(sorter_i,i)
        
        #matrix_out = matrix[sorter_i][:,sorter_j]
        matrix_out = matrix[sorter_i][:,sorter_j]
        
        if return_all:
            return matrix_out, sorter_i, sorter_j, lim
        else:
            return matrix_out
            
    @staticmethod
    def sort_matrix_pop_nans_SIM(matrix):
        sorter_j = np.where(~np.all(np.isnan(matrix),axis=0))[0]
        sorter_i = np.copy(sorter_j)  
        lim = len(sorter_j) - 1

        matrix_out = matrix[sorter_i][:,sorter_j]
            
        return matrix_out, sorter_i, sorter_j, lim
        
            
            
            
    ###################################################
    # COMPOUND PROPERTIES OF STIMULATED-RESPONDING PAIR
    ###################################################
    def get_max_deltaFoverF(self, occ2, time, mode="max_abs", 
                            normalize="baseline",
                            nans_to_zero=False):
        '''Returns the Funatlas-compiled peak deltaF/F in an occ2-style matrix,
        using the fitted responses as de-noised signals.
        
        Parameters
        ----------
        occ2:
            Reference occ2 from get_occurrence_matrix.
        time: numpy.ndarray of floats
            Time axis to use to calculated deltaF from the fitted response.
            
        Returns
        -------
        dFF: occ2-style 
            dFF[i,j] is the list of dFF for the connection i<-j.
        '''
        
        n_neu = len(occ2)
        dFF = np.empty((n_neu,n_neu),dtype=object)
        skip_eval = False
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                dff = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    
                    ec = self.fconn[ds].get_unc_fit_ec(stim, resp_neu_i)
                    if ec is None: 
                        if nans_to_zero: 
                            max_val = 0.0
                            skip_eval = True
                        else: 
                            max_val=np.nan
                            continue #For backcompatibility
                    else:
                        skip_eval = False
                    
                    if mode=="max_abs" and not skip_eval:
                        max_val = np.max(np.abs(ec.eval(time)))
                    elif mode=="avg" and not skip_eval:
                        max_val = np.average(ec.eval(time))
                    if normalize=="baseline":
                        sig = self.sig[ds][:,resp_neu_i]
                        i0 = self.fconn[ds].i0s[stim]
                        shift_vol = self.fconn[ds].shift_vol
                        baseline = np.nanmedian(sig[i0:i0+shift_vol])
                        if baseline==0: 
                            print("Funatlas.get_max_deltaFoverF found 0 baseline (ds,stim,ineu)",ds,stim,ineu)
                            max_val = np.nan
                        else:
                            max_val /= baseline
                    dff.append(max_val)
                dFF[ineu,jneu] = dff
                
        return dFF
        
    def get_deltaFoverF(self, occ2, interval=(0,30), normalize=["baseline"],
                        return_all=False,exclude_ds=[],correct_decaying=True,init=0):
        '''Returns the Funatlas-compiled average deltaF/F over the given 
        interval.
        
        Parameters
        ----------
        occ2:
            Reference occ2 from get_occurrence_matrix.
        interval: array_like (2,) (optional)
            Time interval in seconds. Zero at stimulation time. Default: (0,30)
        normalize: list of str (optional)
            If baseline, regular dF/F. If std, dF/stdev before stim. If
            baseline,std, dF/F/stdev. Default: ["baseline"].
        return_all: bool (optional)
            If True, the function returns also dFF_all, a numpy-array of lists
            containing the individual DF/F whose average are the elements of 
            dFF.
        exclude_ds: array_like (optional)
            Indices of the datasets to skip. Default: [] (no datasets skipped)
        correct_decaying: bool (optional)
            Whether to apply the correction of decaying responses to previous 
            stimulation. Unless you're debugging, use True. Default: True.
                        
        Returns
        -------
        dFF: 2D numpy.ndarray
            dFF[i,j] is the average dFF for the connection i<-j.
        dFF: 2D numpy.ndarray of lists
            dFF_all[i,j][k] is the dFF for the k-th trial recorded for the i<-j
            connection.
        '''
        
        n_neu = len(occ2)
        dFF = np.zeros((n_neu,n_neu))*init
        if return_all:
            dFF_all = np.empty((n_neu,n_neu),dtype=object)
            traces = np.empty((n_neu,n_neu),dtype=object)
            for a in np.ravel(dFF_all): a = []
            for a in np.ravel(traces): a = []
            
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                dff = []
                traces_ = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    if ds in exclude_ds: continue
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    
                    dt0 = int(interval[0]/self.fconn[ds].Dt)
                    dt1 = int(interval[1]/self.fconn[ds].Dt)
                    
                    i0 = self.fconn[ds].i0s[stim]
                    i1 = self.fconn[ds].i1s[stim]
                    shift_vol = self.fconn[ds].shift_vols[stim]
                    Dt = self.fconn[ds].Dt
                    y = self.sig[ds].get_segment(i0,i1,baseline=False,
                                         normalize="none")[:,resp_neu_i]
                    nan_mask = self.sig[ds].get_segment_nan_mask(i0,i1)[:,resp_neu_i]
                    
                    if shift_vol-dt0 > 1:                     
                        std = np.std(y[:shift_vol-dt0])
                    else:
                        std = 0.0
                        
                    if shift_vol > 1:
                        pre = np.average(y[:shift_vol])
                    else:  
                        pre = 0.0
                    
                    #y = y[shift_vol-dt0:shift_vol+dt1+1] - pre
                    
                    if correct_decaying:
                        _,_,_,_,_,df_s_unnorm = self.get_significance_features(
                                    self.sig[ds],resp_neu_i,i0,i1,shift_vol,
                                    Dt,self.nan_th,return_traces=True)
                        if df_s_unnorm is None: continue
                        # This starts from shift_vol, so 
                        if dt0<0: dt0=0
                        y_ = df_s_unnorm[dt0:dt1]
                    else:
                        y_ = y[shift_vol-dt0:shift_vol+dt1] - pre
                    
                    #if len(y)>0:
                    val = np.average(y_)

                    for norm in normalize:
                        if norm=="baseline":
                            if pre !=0.0: 
                                val /= pre
                            else:
                                val = np.inf
                        if norm=="std":
                            val /= std
                    
                    dff.append(val)
                    traces_.append(y_)
                
                if len(dff)>0:
                    adff = np.average(dff)
                    dFF[ineu,jneu] = adff
                    
                if return_all: 
                    dFF_all[ineu,jneu] = dff
                    traces[ineu,jneu] = traces_
        
        if return_all:
            return dFF, dFF_all, traces
        else:
            return dFF
                    
    
    def get_peak_times(self, occ2, time, 
                       use_kernels=False, drop_saturation_branches=False):
        '''Returns the Funatlas-compiled peak times in an occ2-style matrix,
        calling the ExponentialConvolution function for peak times.
        
        Parameters
        ----------
        occ2:
            Reference occ2 from get_occurrence_matrix.
        time: numpy.ndarray of floats
            Time axis to use to find the effective peak time.
        use_kernels: bool (optional)
            Whether to use the kernels or just the fitted responses.
            Default: False.
        drop_saturation_branches: bool (optional)
            Whether to drop the saturation branches from the kernels. Ignored if
            use_kernels is False. Default: False.
            
        Returns
        -------
        peak_times: occ2-style 
            peak_times[i,j] is the list of peak times for the connection
            i<-j.
        '''
        
        n_neu = len(occ2)
        peak_times = np.empty((n_neu,n_neu),dtype=object)
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                pkt = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    
                    if not use_kernels:
                        ec = self.fconn[ds].get_unc_fit_ec(stim, resp_neu_i)
                    else:
                        ec = self.fconn[ds].get_kernel_ec(stim, resp_neu_i)
                        if drop_saturation_branches and ec is not None:
                            ec = ec.drop_saturation_branches()
                    if ec is None: 
                        pkt.append(np.nan)
                        continue
                    
                    peak_time = ec.get_peak_time(time)
                    pkt.append(peak_time)                        
                
                peak_times[ineu,jneu] = np.array(pkt)
        
        return peak_times
        
    def get_min_timescale(self,occ2,time,
                           use_kernels=False, drop_saturation_branches=False,
                           return_kernels=False):
        '''Returns the Funatlas-compiled effective rise times in an occ2-style
        matrix, calling the ExponentialConvolution function for effective rise 
        times.
        
        Parameters
        ----------
        occ2:
            Reference occ2 from get_occurrence_matrix.
        time: numpy.ndarray of floats
            Time axis to use to find the effective rise time.
        use_kernels: bool (optional)
            Whether to use the kernels or just the fitted responses.
            Default: False.
        drop_saturation_branches: bool (optional)
            Whether to drop the saturation branches from the kernels. Ignored if
            use_kernels is False. Default: False.
        return_kernels: bool (optional)
            Whether to return also the kernels. Ignored if use_kernels is False.
            Default: False
            
        Returns
        -------
        min_times: occ2-style 
            rise_times[i,j] is the list of rise times for the connection
            i<-j.
        kernels: occ2-style
            kernels[i,j] is the list of ExponentialConvolution objects
            representing the kernels for the pair i<-j.
        '''
        
        n_neu = len(occ2)
        min_times = np.empty((n_neu,n_neu),dtype=object)
        if return_kernels: kernels = np.empty((n_neu,n_neu),dtype=object)
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                rts = []
                if return_kernels: krnls = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    
                    if not use_kernels:
                        ec = self.fconn[ds].get_unc_fit_ec(stim, resp_neu_i)
                    else:
                        ec = self.fconn[ds].get_kernel_ec(stim, resp_neu_i)
                        if drop_saturation_branches and ec is not None:
                            ec = ec.drop_saturation_branches()
                        if return_kernels: krnls.append(ec)
                    if ec is None: 
                        rts.append(np.nan)
                        continue
                    
                    rt = ec.get_min_time()
                    rts.append(rt)
                
                min_times[ineu,jneu] = np.array(rts)
                if return_kernels: kernels[ineu,jneu] = np.array(krnls)
        
        if return_kernels:
            return min_times, kernels
        else:
            return min_times
    
    def get_eff_rise_times(self,occ2,time,
                           use_kernels=False, drop_saturation_branches=False,
                           return_kernels=False):
        '''Returns the Funatlas-compiled effective rise times in an occ2-style
        matrix, calling the ExponentialConvolution function for effective rise 
        times.
        
        Parameters
        ----------
        occ2:
            Reference occ2 from get_occurrence_matrix.
        time: numpy.ndarray of floats
            Time axis to use to find the effective rise time.
        use_kernels: bool (optional)
            Whether to use the kernels or just the fitted responses.
            Default: False.
        drop_saturation_branches: bool (optional)
            Whether to drop the saturation branches from the kernels. Ignored if
            use_kernels is False. Default: False.
        return_kernels: bool (optional)
            Whether to return also the kernels. Ignored if use_kernels is False.
            Default: False
            
        Returns
        -------
        rise_times: occ2-style 
            rise_times[i,j] is the list of rise times for the connection
            i<-j.
        kernels: occ2-style
            kernels[i,j] is the list of ExponentialConvolution objects
            representing the kernels for the pair i<-j.
        '''
        
        n_neu = len(occ2)
        rise_times = np.empty((n_neu,n_neu),dtype=object)
        if return_kernels: kernels = np.empty((n_neu,n_neu),dtype=object)
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                rts = []
                if return_kernels: krnls = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    
                    if not use_kernels:
                        ec = self.fconn[ds].get_unc_fit_ec(stim, resp_neu_i)
                    else:
                        ec = self.fconn[ds].get_kernel_ec(stim, resp_neu_i)
                        if drop_saturation_branches and ec is not None:
                            ec = ec.drop_saturation_branches()
                        if return_kernels: krnls.append(ec)
                    if ec is None: 
                        rts.append(np.nan)
                        continue
                    
                    rt = ec.get_effective_rise_time(time)
                    rts.append(rt)
                
                rise_times[ineu,jneu] = np.array(rts)
                if return_kernels: kernels[ineu,jneu] = np.array(krnls)
        
        if return_kernels:
            return rise_times, kernels
        else:
            return rise_times
    
    def get_eff_decay_times(self, occ2, time, 
                            use_kernels=False, drop_saturation_branches=False):
        '''Returns the Funatlas-compiled effective decay times in an occ2-style
        matrix, calling the ExponentialConvolution function for effective decay 
        times.
        
        Parameters
        ----------
        occ2:
            Reference occ2 from get_occurrence_matrix.
        time: numpy.ndarray of floats
            Time axis to use to find the effective decay time.
        use_kernels: bool (optional)
            Whether to use the kernels or just the fitted responses.
            Default: False.
        drop_saturation_branches: bool (optional)
            Whether to drop the saturation branches from the kernels. Ignored if
            use_kernels is False. Default: False.
            
        Returns
        -------
        decay_times: occ2-style 
            decay_times[i,j] is the list of decay times for the connection
            i<-j.
        '''
        n_neu = len(occ2)
        decay_times = np.empty((n_neu,n_neu),dtype=object)
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                dts = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    
                    if not use_kernels:
                        ec = self.fconn[ds].get_unc_fit_ec(stim, resp_neu_i)
                    else:
                        ec = self.fconn[ds].get_kernel_ec(stim, resp_neu_i)
                        if drop_saturation_branches and ec is not None:
                            ec = ec.drop_saturation_branches()
                    if ec is None: 
                        dts.append(np.nan)
                        continue
                    
                    dt = ec.get_effective_decay_time(time)
                    dts.append(dt)
            
                decay_times[ineu,jneu] = np.array(dts)
        
        return decay_times
        
    def get_labels_confidences(self, occ2):
        '''Returns the confidences of the labels in an occ2-style matrix.
        
        Parameters
        ----------
        occ2:
            occ2 from get_occurrence_matrix.
            
        Returns
        -------
        conf1, conf2:
            occ1 and occ2 style confidecence.
        '''
        
        n_neu = len(occ2)
        conf1 = np.zeros((n_neu,n_neu))
        conf2 = np.empty((n_neu,n_neu),dtype=object)
        
        m1conf = [[] for i in np.arange(len(self.ds_list))]
        
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                conf = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    stim_neu_j = self.fconn[ds].stim_neurons[stim]
                    
                    lbl = self.labels[ds][resp_neu_i]
                    if len(lbl)<2 or lbl in [""," ","_"]: print("Empty label",lbl,ds,resp_neu_i)
                    
                    conf_i = self.labels_confidences[ds][resp_neu_i]
                    conf_j = self.labels_confidences[ds][stim_neu_j]
                    if stim_neu_j ==-3:
                        # Manually labeled in identify_stim_neuron 
                        conf_j = 1
                                            
                    if conf_i>=0 and conf_j>=0:
                        conf.append(conf_i*conf_j)
                    else:
                        conf.append(0)
                        if conf_i<0:
                            m1conf[ds].append(resp_neu_i)
                        if conf_j<0:
                            m1conf[ds].append(stim_neu_j)
                conf = np.array(conf)
                conf2[ineu,jneu] = conf.copy()
                
                if len(conf)>0:
                    if len(conf[conf>=0])>0:
                        conf1[ineu,jneu] = np.average(conf[conf>=0])
        
        # Diagnostics on confidences not set            
        for ds in np.arange(len(self.ds_list)):
            m1c = m1conf[ds]
            m1c = np.unique(m1c)
            if len(m1c)>0:
                m1cb = m1c[np.where( m1c<len(self.labels[ds]) )]
                m1cc = m1c[np.where( ~(m1c<len(self.labels[ds])) )]
                m1cl = np.array(self.labels[ds])[m1cb]
                print("Funatlas.get_labels_confidences found in",self.ds_list[ds].split("/")[-2],"-1 conf for",m1cl)
                if len(m1cc)>0:
                    print("\tSomething's wrong with",m1cc)
                
        return conf1, conf2
        
    def get_distances(self,occ2):
        '''Returns the average relative distance for stimulated-responding
        pairs of neurons.
        
        Parameters
        ----------
        occ2:
            occ2 from Funatlas.get_occurrence_matrix
        
        Returns
        -------
        dist1, dist2:
            occ1 and occ2-style matrices containing the distances.
        '''
        
        n_neu = len(occ2)
        dist1 = np.ones((n_neu,n_neu))*np.nan
        dist2 = np.empty((n_neu,n_neu),dtype=object)
        for ineu in np.arange(n_neu):
            for jneu in np.arange(n_neu):
                D = []
                for o in occ2[ineu,jneu]:
                    ds = o["ds"]
                    stim = o["stim"]
                    resp_neu_i = o["resp_neu_i"]
                    stim_neu_j = self.fconn[ds].stim_neurons[stim]
                    
                    d = self.brains[ds].get_distance(0,resp_neu_i,stim_neu_j)
                    D.append(d)
            
                dist2[ineu,jneu] = np.array(D)
                dist1[ineu,jneu] = np.average(D)
        
        return dist1,dist2
        
    def get_distances_resp_nonresp(self,req_auto_resp=False):
        '''Returns an array of the distances from the targeted neuron and 
        a corresponding boolean array telling if the neuron responded.
        
        Parameters
        ----------
        req_auto_resp: bool
            Whether to require auto responses.
        
        Returns
        -------
        dist3: 1D numpy.ndarray of floats
            Array of the distances in pixel values (raveled).
        dist4: 2D numpy.ndarray of lists of floats
            Distances in dist3 still split up by [ds][ie,ineu].
        resp3: 1D numpy.ndarray of bools
            Whether the neuron at the corresponding element in dist3 was a
            responding neuron.
        resp4:
            resp3 still split up by [ds][ie,ineu].
        '''
        
        dist3 = np.zeros(0)
        dist4 = []
        resp3 = np.zeros(0,dtype=bool)
        resp4 = []
        for ds in np.arange(len(self.ds_list)):
            # Initialize 2D arrays containing the distances between stimulated
            # neuron and other neurons for each stimulation and corresponding
            # boolean 2D array saying whether that neuron responded to that
            # stimulation.
            dist_ = np.zeros((self.fconn[ds].n_stim,self.fconn[ds].n_neurons))
            resp_ = np.zeros((self.fconn[ds].n_stim,self.fconn[ds].n_neurons),dtype=bool)
            for ie in np.arange(self.fconn[ds].n_stim):
                # Determine if the stimulated neuron responded.
                auto_resp = self.fconn[ds].stim_neurons[ie] in self.fconn[ds].resp_neurons_by_stim[ie]
                if self.fconn[ds].stim_neurons[ie]<0 or (not auto_resp and req_auto_resp): 
                    dist_[ie] = np.nan
                else:
                    resp_[ie][self.fconn[ds].resp_neurons_by_stim[ie]] = True
                    j_neu = self.fconn[ds].stim_neurons[ie]
                    for i_neu in np.arange(self.fconn[ds].n_neurons):
                        d = self.brains[ds].get_distance(0,i_neu,j_neu)
                        dist_[ie][i_neu] = d
            dist3 = np.append(dist3,np.ravel(dist_))
            dist4.append(dist_)
            resp3 = np.append(resp3,np.ravel(resp_))
            resp4.append(resp_)
        
        return dist3,dist4,resp3,resp4
        
    def get_distances_from_conn(self,c):
        '''Compiles the distances between stimulated and other neurons assuming
        the responding neurons are dictated by the connectome c.
        
        Parameters
        ----------
        c: 2D numpy.ndarray of bools
            Boolean connectome.
        
        Returns
        -------
        '''
        
        dist3 = np.zeros(0)
        dist4 = []
        resp3 = np.zeros(0,dtype=bool)
        resp4 = []
        for ds in np.arange(len(self.ds_list)):
            dist_ = np.zeros((self.fconn[ds].n_stim,self.fconn[ds].n_neurons))
            resp_ = np.zeros((self.fconn[ds].n_stim,self.fconn[ds].n_neurons),dtype=bool)
            
            # Get the labels of the neurons in the dataset, to then obtain
            # the atlas indices of the responding neurons.
            lbls = self.brains[ds].get_labels(0)
            for ie in np.arange(self.fconn[ds].n_stim):
                if self.stim_neurons_ai[ds][ie]<0: 
                    dist_[ie] = np.nan
                else:
                    # Get atlas and dataset indices of the stimulated neuron.
                    aj_neu = self.stim_neurons_ai[ds][ie]
                    j_neu = self.fconn[ds].stim_neurons[ie]
                    for i_neu in np.arange(self.fconn[ds].n_neurons):
                        if lbls[i_neu]!="":
                            # Get the atlas index of the responding neuron
                            ai_neu = self.ids_to_i(lbls[i_neu])
                            # Set the response of the dataset-index element
                            # to the boolean connectome (indexed with atlas
                            # indices).
                            resp_[ie][i_neu] = c[ai_neu,aj_neu]
                            # Compute the distance using the dataset indices.
                            d = self.brains[ds].get_distance(0,i_neu,j_neu)
                            dist_[ie][i_neu] = d
                        else:
                            dist_[ie][i_neu] = np.nan
                            
            dist3 = np.append(dist3,np.ravel(dist_))
            dist4.append(dist_)
            resp3 = np.append(resp3,np.ravel(resp_))
            resp4.append(resp_)
        
        return dist3,dist4,resp3,resp4
        
    @staticmethod
    def weighted_avg_occ2style(A, w, return_rele=False):
        '''Given occ2-style objects A and w, the function returns the average 
        of the array contained in A[i,j] weigthed by the array contained in
        w[i,j]. For example, A can be the result of get_eff_rise_times() and
        w can be the result of get_max_deltaFoverF().
        
        Parameters
        ----------
        A: 2d numpy.ndarray of numpy.ndarrays of floats
            A[i,j] are the arrays to be averaged.
        w: 2d numpy.ndarray of numpy.ndarrays of floats
            w[i,j] are the weights for the array A[i,j]. Must have the same 
            shape as A.
            
        Returns
        -------
        B: 2d numpy.ndarray of floats
            B[i,j] is the weighted average of the array in A[i,j]. B[i,j] is 
            nan if A[i,j] is empty.
        '''
        
        n = A.shape[0]
        m = A.shape[1]
        
        B = np.zeros((n,m))*np.nan
        rele = np.zeros((n,m))*np.nan
        
        for i in np.arange(n):
            for j in np.arange(m):
                if len(A[i,j])>0 and np.nansum(w[i,j])>0:
                    B[i,j] = np.nansum(A[i,j]*w[i,j])/np.nansum(w[i,j])
                    if return_rele:
                        rele[i,j] = np.sqrt(np.average(np.power(A[i,j]-B[i,j],2),weights=w[i,j]))
                        rele[i,j] /= B[i,j]
                        if rele[i,j] == 0: rele[i,j] = np.nan
        
        if return_rele:
            return B, rele
        else:
            return B
            
    @staticmethod
    def weighted_avg_occ2style2(A, ws, return_rele=False, abs_w=True):
        '''Given occ2-style objects A and w, the function returns the average 
        of the array contained in A[i,j] weigthed by the array contained in
        w[i,j]. For example, A can be the result of get_eff_rise_times() and
        w can be the result of get_max_deltaFoverF().
        
        Parameters
        ----------
        A: 2d numpy.ndarray of numpy.ndarrays of floats
            A[i,j] are the arrays to be averaged.
        ws: array_like of 2d numpy.ndarray of numpy.ndarrays of floats
            w[k][i,j] are the weights for the array A[i,j]. Must have the same 
            shape as A. Absolute values will be taken unless abs_w is
            False.
        abs_w: bool (optional)
            Whether to take the absolute values of the weights. Default: True.
            
        Returns
        -------
        B: 2d numpy.ndarray of floats
            B[i,j] is the weighted average of the array in A[i,j]. B[i,j] is 
            nan if A[i,j] is empty.
        '''
        n = A.shape[0]
        m = A.shape[1]
        
        B = np.zeros((n,m))*np.nan
        rele = np.zeros((n,m))*np.nan
        
        for i in np.arange(n):
            for j in np.arange(m):
                
                # Element-wise multiplication of corresponding weights from
                # the different sets of weights
                w = np.ones(len(ws[0][i,j]))
                for iw in np.arange(len(ws)):
                    if abs_w:
                        w *= np.absolute(np.array(ws[iw][i,j]))
                    else:
                        w *= np.array(ws[iw][i,j])
                        
                # Weighted average with the processed weights.
                if len(A[i,j])>0 and np.nansum(w)>0 and np.nansum(A[i,j])>0:
                    B[i,j] = np.nansum(A[i,j]*w)/np.nansum(w)
                    if return_rele:
                        rele2 = np.average(np.power(A[i,j]-B[i,j],2),weights=w)
                        if not np.isnan(rele2) and not np.isinf(rele2):
                            rele[i,j] = np.sqrt(rele2)
                            rele[i,j] /= B[i,j]
                        else:
                            rele[i,j] = np.nan
                        if rele[i,j] == 0: rele[i,j] = np.nan
        
        if return_rele:
            return B, rele
        else:
            return B
            
    @staticmethod
    def filter_occ2(A,f,leq=None,geq=None):
        '''Filters the occ2-style matrix A based on a condition on a similarly  
        structured matrix f. The condition can be <= or =>. The function returns
        the elements of A sorted into two occ2-style matrices corresponding to
        True and False conditions.
        
        Parameters
        ----------
        A: 2D numpy.ndarray of lists
            The matrix to be filtered.
        f: 2D numpy.ndarray of lists
            The matrix on which to set the conditions. Must have the same
            structure as A.
        leq: float (optional)
            If not None, the condition is f<=leq. Default: None.
        geq: float (optional)
            If not None, the condition is f>=leq. Default: None.
            
        Returns
        -------
        A1: 2D numpy.ndarray of lists
            The elements of A that satisfy the condition.
        A2: 2D numpy.ndarray of lists
            The elements of A that do not satisfy the condition.
        '''
        
        if (leq is None and geq is None) or (leq is not None and geq is not None):
            raise ValueError("Funatlas.filter_occ2() one of the parameters leq and geq must not be None")
        
        n,m = A.shape
        A1 = np.empty_like(A) # The elements for which the condition is True
        A2 = np.empty_like(A) # The elements for which the condition is False
        for i in np.arange(n):
            for j in np.arange(m):
                toA1 = []
                toA2 = []
                for io in np.arange(len(A[i,j])):
                    if not (np.isnan(f[i,j][io]) or np.isinf(f[i,j][io])):
                        if leq is not None:
                            if f[i,j][io]<=leq:
                                toA1.append(io)
                            else:
                                toA2.append(io)
                        elif geq is not None:
                            if f[i,j][io]>=geq:
                                toA1.append(io)
                            else:
                                toA2.append(io)
                A1[i,j] = [A[i,j][io] for io in toA1]
                A2[i,j] = [A[i,j][io] for io in toA2]

        return A1, A2
        
    @staticmethod
    def regenerate_occ1(occ2):
        n,m = occ2.shape
        A = np.zeros((n,m),dtype=int)
        for i in np.arange(n):
            for j in np.arange(m):
                A[i,j] = len(occ2[i,j])   
                
        return A
    
    @staticmethod
    def take_winner(occ2a,occ2b,def_winner="a",return_occ1=True):
        '''occ2a wins if there is an equal number of elements in occ2a[i,j] and
        occ2b[i,j].
        '''
        n,m=occ2a.shape
        
        occ2a_out = np.empty((n,m),dtype=object)
        occ2b_out = np.empty((n,m),dtype=object)
        for i in np.arange(n):
            for j in np.arange(m):
                occ2a_out[i,j] = []
                occ2b_out[i,j] = []
        
        for i in np.arange(n):
            for j in np.arange(m):
                if occ2b_out[i,j] is None: print("bb")
                if len(occ2a[i,j])>len(occ2b[i,j]):
                    occ2a_out[i,j] = occ2a[i,j]
                elif len(occ2a[i,j])<len(occ2b[i,j]):
                    occ2b_out[i,j] = occ2b[i,j]
                else:
                    if def_winner=="a":
                        occ2a_out[i,j] = occ2a[i,j]
                    else:
                        occ2b_out[i,j] = occ2b[i,j]
        if return_occ1:
            occ1a_out = Funatlas.regenerate_occ1(occ2a_out)
            occ1b_out = Funatlas.regenerate_occ1(occ2b_out)
            return occ1a_out, occ2a_out, occ1b_out, occ2b_out
        else:    
            return occ2a_out, occ2b_out
                
        
    def filter_occ12_from_sysargv(self,occ2,sysargv,return_all=False):
        '''Utility function that checks if the sys.argv is requesting to filter
        occ2 (and whether on kernel timescales), and returns the filtered occ1
        and occ2.
        
        Parameters
        ----------
        sysargv: list of strings
            The sys.argv. This function will look for one (only one) of
                --leq-rise-time:int
                --geq-rise-time:int
                --leq-decay-time:int
                --geq-decay-time:int
            and
                --use-kernels
        return_all: bool (optional)
            Whether to return also the occ2 corresponding to the False 
            condition. Default: False.
        
        Returns
        -------
        occ1, occ2: filtered
        
        '''
        leq_rise_time = np.nan
        geq_rise_time = np.nan
        leq_peak_time = np.nan
        geq_peak_time = np.nan
        use_kernels = "--use-kernels" in sysargv
        drop_saturation_branches = "--drop-saturation-branches" in sysargv
        
        for s in sysargv:
            sa = s.split(":")
            if sa[0] == "--leq-rise-time": leq_rise_time = float(sa[1])
            if sa[0] == "--geq-rise-time": geq_rise_time = float(sa[1])
            if sa[0] == "--leq-decay-time": leq_decay_time = float(sa[1])
            if sa[0] == "--geq-decay-time": geq_decay_time = float(sa[1])
        
        # Determine if occ2 needs to be filtered
        whr=np.where(~np.isnan([leq_rise_time,geq_rise_time,leq_peak_time,geq_peak_time]))[0]
        if len(whr)>1:
            raise ValueError("Too many leq geq conditions.")
        elif len(whr)==1:
            # Make time axis to find rise or decay times
            if not use_kernels: time2 = np.linspace(0,200,1000)
            else: time2 = np.linspace(0,10,1000)
                
            if not np.isnan(leq_rise_time):
                rise_times = self.get_eff_rise_times(
                        occ2,time2,use_kernels=use_kernels,
                        drop_saturation_branches=drop_saturation_branches)
                occ2true, occ2false = self.filter_occ2(
                                        occ2,rise_times,leq=leq_rise_time)
            elif not np.isnan(geq_rise_time):
                rise_times = self.get_eff_rise_times(
                        occ2,time2,use_kernels=use_kernels,
                        drop_saturation_branches=drop_saturation_branches)
                occ2true, occ2false  = self.filter_occ2(
                                        occ2,rise_times,geq=geq_rise_time)
            elif not np.isnan(leq_decay_times):
                decay_times = self.get_eff_decay_times(
                        occ2,time2,use_kernels=use_kernels,
                        drop_saturation_branches=drop_saturation_branches)
                occ2true, occ2false  = self.filter_occ2(
                                        occ2,rise_times,leq=leq_decay_time)
            elif not np.isnan(leq_decay_times):
                decay_times = self.get_eff_decay_times(
                        occ2,time2,use_kernels=use_kernels,
                        drop_saturation_branches=drop_saturation_branches)
                occ2true, occ2false  = self.filter_occ2(
                                        occ2,rise_times,geq=geq_decay_time)
        
            occ1true = self.regenerate_occ1(occ2true)
            occ1false = self.regenerate_occ1(occ2false)
            
            if return_all:
                return occ1true,occ2true,occ1false,occ2false
            else:
                return occ1true,occ2true
        else:
            occ1 = self.regenerate_occ1(occ2)
            if return_all:
                return occ1,occ2,None,None
            else:
                return occ1,occ2
    
    @staticmethod
    def ravel_occ2(A):
        Ar = []
        for a in np.ravel(A):
            if a is not None:
                for a_ in a: Ar.append(a_)
                
        return np.array(Ar)
        
    @staticmethod
    def average_occ2(A):
        n,m = A.shape
        B = np.zeros((n,m))*np.nan
        for i in np.arange(n):
            for j in np.arange(m):
                if len(A[i,j])>0:
                    B[i,j] = np.average(A[i,j])
        
        return B
        
    @staticmethod
    def median_occ2(A):
        n,m = A.shape
        B = np.zeros((n,m))*np.nan
        for i in np.arange(n):
            for j in np.arange(m):
                if len(A[i,j])>0:
                    B[i,j] = np.median(A[i,j])
        
        return B
    
    @staticmethod 
    def std_occ2(A):
        n,m = A.shape
        B = np.zeros((n,m))*np.nan
        for i in np.arange(n):
            for j in np.arange(m):
                if len(A[i,j])>0:
                    B[i,j] = np.std(A[i,j])
        
        return B
                    
        
    def get_qvalues2(self,merge_bilateral,req_auto_response):
        folder = "/projects/LEIFER/Sophie/Figures/Response_Statistics/"
        
        mb = "T" if merge_bilateral else "F"
        raq = "T" if req_auto_response else "F"
        fname = "qvalues_matrix_merge"+mb+"_autoresp"+raq+".pickle"
        
        f = open(folder+fname,"rb")
        q = pickle.load(f)
        f.close()
        
        return q
        
    def get_qvalues(self, occ1, occ3, exclude_zero_k):
        '''Sophie'''
        folders = np.loadtxt("/projects/LEIFER/Sophie/Lists/No_stim_list.txt", dtype='str', delimiter="\n")
        delta_t_pre = 30.0
        ampl_min_time_og = 4. 
        ampl_thresh_og = 1.3
        deriv_min_time_og = 1.
        deriv_thresh_og = 1. 
        nan_thresh_og = 0.5
        smooth_mode = "sg_causal" #CHANGED
        smooth_n = 13 #CHANGED
        smooth_poly = 1
        resp_neurons_by_stim = {}
        number_of_neurons = {}
        
        cache_file = "/projects/LEIFER/francesco/funatlas/qvalue_p.json"
        calc_everything = True
        if os.path.isfile(cache_file):
            f = open(cache_file,"r")
            jc = json.load(f)
            f.close()
            
            if ampl_thresh_og==jc["ampl_thresh_og"] and ampl_min_time_og==jc["ampl_min_time_og"] \
               and deriv_thresh_og==jc["deriv_thresh_og"] and deriv_min_time_og==jc["deriv_min_time_og"]\
               and nan_thresh_og==jc["nan_thresh_og"]:
               p = jc["p"]
               calc_everything = False
               
        if calc_everything:   
            for f in range(len(folders)):
                # Get the recording object
                rec_original = wormdm.data.recording(folders[f])
                shift_vol = int(delta_t_pre / rec_original.Dt)
                events = rec_original.get_events()
                # Load the signal
                sig = wormdm.signal.Signal.from_file(folders[f], "green")
                sig.appl_photobl()
                sig.remove_spikes()
                # Smooth and calculate the derivative of the signal (derivative needed for
                # detection of responses)
                sig.smooth(n=smooth_n, i=None, poly=smooth_poly, mode=smooth_mode)
                sig.derivative = sig.get_derivative(sig.unsmoothed_data, 11, 1)

                # Get the neurons coordinates of the reference volume and load the matches
                # to determine what neuron was targeted

                cervelli = wormb.Brains.from_file(folders[f])
                cervelli.load_matches(folders[f])

                nr_of_shifts = 2
                for s in np.arange(nr_of_shifts):
                    shift_by = (rec_original.optogeneticsFrameCount[3] - rec_original.optogeneticsFrameCount[2]) / nr_of_shifts
                    # Create functional connectome
                    rec = wormdm.data.recording(folders[f])
                    rec.optogeneticsFrameCount = rec_original.optogeneticsFrameCount + (s * shift_by)
                    fconn = pp.Fconn.from_objects(
                        rec, cervelli, sig, delta_t_pre,verbose=False,
                        nan_thresh=nan_thresh_og, deriv_thresh=deriv_thresh_og, ampl_thresh=ampl_thresh_og,
                        deriv_min_time=deriv_min_time_og, ampl_min_time=ampl_min_time_og)
                    
                    #'''print(ampl_thresh_og,fconn.ampl_thresh)
                    '''assert ampl_thresh_og == fconn.ampl_thresh, "Amplitude Threshold No Longer Same"
                    assert ampl_min_time_og == fconn.ampl_min_time, "Amplitude Min Time No Longer Same"
                    assert deriv_thresh_og == fconn.deriv_thresh, "Derivative Threshold No Longer Same"
                    assert deriv_min_time_og ==  fconn.deriv_min_time, "Derivative Min Time No Longer Same"
                    assert nan_thresh_og == fconn.nan_thresh'''

                    n_neurons = sig.data.shape[1]
                    number_of_neurons[str(f)] = n_neurons
                    resp_neurons_by_stim[str(f) + str(s)] = fconn.resp_neurons_by_stim


            total_could_resp = 0
            total_respond = 0
            probability_by_dataset = {}
            for f in np.arange(len(folders)):
                total_could_resp_f = 0
                total_respond_f = 0
                for s in np.arange(nr_of_shifts):
                    nr_stims_folder = len(resp_neurons_by_stim[str(f) + str(s)])
                    for ie in np.arange(nr_stims_folder):
                        total_could_resp = total_could_resp + number_of_neurons[str(f)]
                        total_respond = total_respond + len(resp_neurons_by_stim[str(f) + str(s)][ie])
                        total_could_resp_f = total_could_resp_f + number_of_neurons[str(f)]
                        total_respond_f = total_respond_f + len(resp_neurons_by_stim[str(f) + str(s)][ie])

                probability_by_dataset[str(f)] = total_respond_f / total_could_resp_f
            probability_of_resp = total_respond / total_could_resp

            iids = jids = self.neuron_ids  # gives ids for each global index

            pvalue_matrix = np.empty((len(iids), len(jids)))
            pvalue_matrix[:] = np.NaN

            p = probability_of_resp
            
            f = open(cache_file,"w")
            json.dump({"ampl_thresh_og":ampl_thresh_og,"ampl_min_time_og":ampl_min_time_og,
                       "deriv_thresh_og":deriv_thresh_og,"deriv_min_time_og":deriv_min_time_og,
                       "nan_thresh_og":nan_thresh_og,"p":p},f)
            f.close()
            os.chmod(cache_file,0o775)
        
        iids = jids = self.neuron_ids  # gives ids for each global index
        pvalue_matrix = np.empty((len(iids), len(jids)))
        pvalue_matrix[:] = np.NaN
        
        for iid in iids:
            for jid in jids:
                
                # TODO THIS IS NOT NECESSARY BECAUSE iids AND jids ARE ALREADY
                # FROM neuron_ids. ids_to_i RETURNS THE INDEX OF iid AND jid IN
                # neuron_ids.
                # Convert the requested IDs to atlas-indices.
                i, j = self.ids_to_i([iid, jid])
                if i < 0: print(iid, "not found. Check approximations.")
                if j < 0: print(jid, "not found. Check approximations.")

                n = occ3[i, j]
                k = occ1[i, j]
                if k != 0 and exclude_zero_k:
                    pvalue = 1 - binom.cdf(k - 1, n, p)
                    pvalue_matrix[i, j] = pvalue

                elif n != 0 and not exclude_zero_k:
                    pvalue = 1 - binom.cdf(k - 1, n, p)
                    pvalue_matrix[i, j] = pvalue

        flat_pvalues = np.matrix.flatten(pvalue_matrix)
        flat_pvalues = flat_pvalues[~np.isnan(flat_pvalues)]
        # TODO I think the following line does not do anything, because it does
        # not act in place. You should use flat_pvalues = flat_pvalues.tolist()
        # But given that it doesn't give you errors, it probably does not matter,
        # qvalue is likely able to use numpy arrays.
        flat_pvalues.tolist()
        _, qvals = qvalue(flat_pvalues)

        qvalues_matrix = np.empty((len(iids), len(jids)))
        qvalues_matrix[:] = np.NaN
        flat_pval_index = 0
        for i in np.arange(len(iids)):
            for j in np.arange(len(jids)):
                pval_current = pvalue_matrix[i, j]
                if not np.isnan(pval_current):
                    #pval_index = np.where(flat_pvalues == pval_current)
                    qvalues_matrix[i, j] = qvals[flat_pval_index]
                    flat_pval_index = flat_pval_index + 1
                    
        return qvalues_matrix
    
    @staticmethod
    def get_significance_features(sig,neu_i,i0,i1,shift_vol,Dt,nan_th,
                                  sderker=None,return_traces=False):
        correct_decaying = True                

        if sderker is None:
            sderker = savgol_coeffs(39, 2, deriv=2, delta=Dt)
        sd = np.convolve(sderker,sig.data[:,neu_i],mode="same")
        if correct_decaying:
            fderker = savgol_coeffs(39, 1, pos=38, deriv=1, delta=Dt)
            smoothker = savgol_coeffs(39, 1, deriv=0, delta=Dt)
            fd = np.convolve(fderker,sig.data[:,neu_i],mode="same")
            smoothy = np.convolve(smoothker,sig.data[:,neu_i],mode="same")
        
        y = sig.get_segment(i0,i1,baseline=False,normalize="none")
        nan_mask = sig.get_segment_nan_mask(i0,i1)
        
        #if np.sum(nan_mask[:,neu_i])>nan_th*len(y[:,neu_i]):
        if not pp.Fconn.nan_ok(nan_mask[:,neu_i],nan_th*len(y[:,neu_i])):
            if return_traces:
                return None,None,None, None, None, None
            else:
                return None,None,None
        else:
            baseline = np.average(y[:shift_vol,neu_i])
            #pre = baseline
            pre = sig.get_loc_std(y[:,neu_i],8)
            if pre==0.0:
                if return_traces:
                    return None,None,None, None, None, None
                else:
                    return None,None,None
            else:
                dytrace_unnorm = (y[shift_vol:shift_vol+60,neu_i]-baseline )
                dytrace = dytrace_unnorm/pre #This was not bound to :+60
                dy = np.average(dytrace) 
                sd__ = np.average(sd[i0+shift_vol:i0+shift_vol+30]) #-5+11
                
                # Correct dy if it is decaying
                if correct_decaying:
                    fd_0 = fd[i0+shift_vol]
                    if fd_0<0: #dy<0 and 
                        if i0+shift_vol+60<smoothy.shape[0]:
                            zero = smoothy[i0+shift_vol+60]
                            length_remaining = 60
                        else:
                            zero = smoothy[-1]
                            length_remaining = smoothy.shape[0] - (i0+shift_vol)
                        time = np.arange(length_remaining)*Dt 
                        A = smoothy[i0+shift_vol]-zero
                        corry = A*np.exp(fd_0*time)+zero
                        dytrace_unnorm = (y[shift_vol:shift_vol+length_remaining,neu_i] - corry )
                        dytrace = dytrace_unnorm/pre
                        dy = np.average(dytrace)
                    
        
        if return_traces:
            return dy,dy,sd__,\
                    dytrace,sd[i0+shift_vol:i0+shift_vol+30],dytrace_unnorm
        else:
            return dy,dy,sd__
            
            
    @classmethod
    def get_ctrl_distributions(cls,strain=""):
        delta_t_pre = 30.0
        ampl_min_time_og = 4. 
        ampl_thresh_og = 1.3
        deriv_min_time_og = 1.
        deriv_thresh_og = 1. 
        nan_thresh_og = 0.5
        
        if strain == "":
            ds_list = "/projects/LEIFER/Sophie/Lists/No_stim_list.txt"
            cache_file = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_params.json"
            cache_file_dff = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_dff.txt"
            cache_file_dff2 = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_dff2.txt"
            cache_file_sd = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_sd.txt"
        elif strain =="unc31":
            print("Funaltas: Using unc31 ctrl measurement")
            ds_list = "/projects/LEIFER/francesco/funatlas_unc31_ctrl_list.txt"
            cache_file = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_params_unc31.json"
            cache_file_dff = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_dff_unc31.txt"
            cache_file_dff2 = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_dff2_unc31.txt"
            cache_file_sd = "/projects/LEIFER/francesco/funatlas/qvalue_kolmogorov_smirnov_sd_unc31.txt"
            
            
        folders = np.loadtxt(ds_list, dtype='str')#, delimiter="\n")
        n_ds = len(folders)

        signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                         "smooth_mode": "sg_causal", 
                         "smooth_n": 13, "smooth_poly": 1,       
                         "photobl_appl":True,          
                         "matchless_nan_th_from_file": True}

        nan_th = cls.nan_th
        cache_dict = {"delta_t_pre": delta_t_pre,
                      "ampl_thresh_og":ampl_thresh_og,
                      "ampl_min_time_og":ampl_min_time_og,
                      "deriv_thresh_og":deriv_thresh_og,
                      "deriv_min_time_og":deriv_min_time_og,
                      "nan_thresh_og":nan_thresh_og,
                      "folders":folders.tolist(),
                      "signal_kwargs":signal_kwargs,
                      "nan_th":nan_th}
        
        load_cached_distributions = False
        if os.path.isfile(cache_file):
            f = open(cache_file,"r")
            cached_p = json.load(f)
            f.close()
            
            all_same = True
            for k in cache_dict.keys():
                if type(cache_dict[k])==dict:
                    for kp in cache_dict[k].keys():
                        if cache_dict[k][kp]!=cached_p[k][kp]:
                            all_same=False 
                elif cache_dict[k]!=cached_p[k]: 
                    all_same=False
            if all_same:
                load_cached_distributions = True
                
        print("USING DF/SIGMA")
        
        if load_cached_distributions:
            print("p values: Loading cached distributions of ctrl measurements")
            dff = np.loadtxt(cache_file_dff)
            dff2 = np.loadtxt(cache_file_dff2)
            sd = np.loadtxt(cache_file_sd)
            
            #plt.hist(dff,bins=100)
            #plt.show()
            #quit()
        
        else:
            dff = np.zeros(0)
            dff2 = np.zeros(0)
            sd = np.zeros(0)
            print("Funatlas n_ds",n_ds)
            
            for i_ds in np.arange(n_ds):
                folder = "/".join(folders[i_ds].split("/")[:-1])+"/"
                fconn = pp.Fconn.from_file(folder)
                sig = wormdm.signal.Signal.from_file(folder,"green",**signal_kwargs)
                print("Funatlas got fconn and sig ",i_ds)
                print(fconn.n_stim)
                
                #sderker = savgol_coeffs(13, 2, deriv=2, delta=fconn.Dt)
                #sder = np.zeros_like(sig.data)            
                #for k in np.arange(sig.data.shape[1]):
                #    sder[:,k] = np.convolve(sderker,sig.data[:,k],mode="same")
                
                dff_ = np.ones((fconn.n_stim,fconn.n_neurons))*np.nan
                dff2_ = np.ones((fconn.n_stim,fconn.n_neurons))*np.nan
                sd_ = np.ones((fconn.n_stim,fconn.n_neurons))*np.nan
                
                for ie in np.arange(fconn.n_stim):
                    i0 = fconn.i0s[ie]
                    i1 = fconn.i1s[ie]
                    shift_vol = fconn.shift_vol
                    #y = sig.get_segment(i0,i1,baseline=False,normalize="none")
                    #nan_mask = sig.get_segment_nan_mask(i0,i1)
                    
                    for ineu in np.arange(fconn.n_neurons):
                        dff__,dff2__,sd__ = cls.get_significance_features(
                                                sig,ineu,i0,i1,shift_vol,
                                                fconn.Dt,nan_th)
                        if dff__ is not None:
                            dff_[ie,ineu] = dff__
                            dff2_[ie,ineu] = dff__
                            sd_[ie,ineu] = sd__
                        else:
                            continue
                            
                dff = np.append(dff,np.ravel(dff_[np.where(~np.isnan(dff_))]))
                dff2 = np.append(dff2,np.ravel(dff2_[np.where(~np.isnan(dff2_))]))
                sd = np.append(sd,np.ravel(sd_[np.where(~np.isnan(sd_))]))
            
            #plt.hist(dff,bins=100)
            #plt.show()
            #quit()
            
            np.savetxt(cache_file_dff,dff)
            np.savetxt(cache_file_dff2,dff2)
            np.savetxt(cache_file_sd,sd)
            f = open(cache_file,"w")
            json.dump(cache_dict,f)
            f.close()
            
        return dff,dff2,sd
        
    @classmethod
    def get_ctrl_cdf(cls,**kwargs):
        dff,dff2,sd = cls.get_ctrl_distributions(**kwargs)
        
        xdff = np.sort(dff)
        xsd = np.sort(sd)
        cdf = np.arange(dff.shape[0])/dff.shape[0]
        
        return xdff, xsd, cdf
        
    @classmethod
    def get_individual_p(cls,values,x,cdf):
        try: 
            len(values)
            was_scalar = False
        except: 
            values = np.array([values])
            was_scalar = True
        
        a = np.interp(values,x,cdf)
        p_ = np.empty((2,len(a)))
        p_[0] = 1-a
        p_[1] = a
        p = np.min(p_,axis=0)
        
        if was_scalar: p=p[0]
        
        return p
        
        
    def get_kolmogorov_smirnov_p(self,inclall_occ2,nan_th=None,
                                 return_tost=False,tost_th=None,
                                 **kwargs):
        if nan_th is None: nan_th = self.nan_th
        ctrl_dff, ctrl_dff2, ctrl_sd = self.get_ctrl_distributions(**kwargs)
        
        if return_tost:
            # Build the individual thresholds for tost given the tost_low and
            std_ctrl_dff = np.std(ctrl_dff)
            std_ctrl_sd = np.std(ctrl_sd)
            tost1_low = -std_ctrl_dff*tost_th
            tost1_upp = std_ctrl_dff*tost_th
            tost2_low = -std_ctrl_sd*tost_th
            tost2_upp = std_ctrl_sd*tost_th
            
        
        
        n_neu = inclall_occ2.shape[0]
        
        n_neu = self.n_neurons
        pdff = np.ones((n_neu,n_neu))*np.nan
        psd = np.ones((n_neu,n_neu))*np.nan
        p = np.ones((n_neu,n_neu))*np.nan
        
        if return_tost:
            tost_pdff = np.ones((n_neu,n_neu))*np.nan
            tost_psd = np.ones((n_neu,n_neu))*np.nan
            tost_p = np.ones((n_neu,n_neu))*np.nan
        
        Dts = []
        for i_ds in np.arange(len(self.ds_list)):
            Dts.append(self.fconn[i_ds].Dt)
            
        all_Dt_equal = np.all(np.diff(Dts)==0)
        
        if all_Dt_equal:
            Dt = self.fconn[0].Dt
            sderker = savgol_coeffs(39, 2, deriv=2, delta=Dt)
        else:
            sderker = None
        
        for aj in np.arange(n_neu):
            for ai in np.arange(n_neu):
                dff_ = []
                sd_ = []
                for occ in inclall_occ2[ai,aj]:
                    ds = occ["ds"]
                    ie = occ["stim"]
                    i = occ["resp_neu_i"]
                    
                    i0 = self.fconn[ds].i0s[ie]
                    i1 = self.fconn[ds].i1s[ie]
                    shift_vol = self.fconn[ds].shift_vol
                    dff__,_,sd__ = self.get_significance_features(
                                                self.sig[ds],i,i0,i1,shift_vol,
                                                self.fconn[ds].Dt,nan_th,
                                                sderker=sderker)
                    if dff__ is not None:
                        dff_.append(dff__)
                        sd_.append(sd__)
                    else:
                        continue
                        
                if len(dff_)>0: 
                    _,pdff[ai,aj] = kstest(ctrl_dff,dff_,alternative="two-sided")
                    _,psd[ai,aj] = kstest(ctrl_sd,sd_,alternative="two-sided")
                    if pdff[ai,aj]==0: pdff[ai,aj]=1e-10
                    if psd[ai,aj]==0: psd[ai,aj]=1e-10
                    _, p[ai,aj] = combine_pvalues([pdff[ai,aj],psd[ai,aj]],
                                                  method="fisher")
                                                  
                    if return_tost:
                        tost_pdff[ai,aj],_,_ = ttost_ind(ctrl_dff,dff_,tost1_low,tost1_upp)
                        tost_psd[ai,aj],_,_ = ttost_ind(ctrl_sd,sd_,tost2_low,tost2_upp)
                        if tost_pdff[ai,aj]==0: tost_pdff[ai,aj]=1e-10
                        if tost_psd[ai,aj]==0: tost_psd[ai,aj]=1e-10
                        _, tost_p[ai,aj] = combine_pvalues([tost_pdff[ai,aj],tost_psd[ai,aj]],
                                                      method="fisher")
                        
        if not return_tost:
            return p, pdff, psd
        else:
            return p, pdff, psd, tost_p, tost_pdff, tost_psd
        
    def get_kolmogorov_smirnov_q(self,*args,**kwargs):
        if "return_p" in kwargs.keys():
            return_p = True
            kwargs.pop("return_p")
        else:
            return_p = False
            
        return_tost = False
        if "return_tost" in kwargs.keys():
            return_tost = kwargs["return_tost"]
                
        if return_tost:
            pvalues,_,_,tost_pvalues,_,_ = self.get_kolmogorov_smirnov_p(*args,**kwargs)
        else:
            pvalues,_,_ = self.get_kolmogorov_smirnov_p(*args,**kwargs)
        
        print("pvalues[pvalues==0] = 1e-10")
        pvalues[pvalues==0] = 1e-10
        
        _, q = fdrqvalue(pvalues[np.isfinite(pvalues)]) 
        q_mat = np.ones_like(pvalues)*np.nan
        q_mat[np.isfinite(pvalues)] = q
        
        if return_tost:
            tost_pvalues[pvalues==0] = 1e-10
            _, tost_q = fdrqvalue(tost_pvalues[np.isfinite(tost_pvalues)]) 
            tost_q_mat = np.ones_like(tost_pvalues)*np.nan
            tost_q_mat[np.isfinite(tost_pvalues)] = tost_q
        
        if return_p and return_tost:
            return q_mat, pvalues, tost_q_mat, tost_pvalues
        elif return_p and not return_tost:
            return q_mat, pvalues
        elif not return_p and return_tost:
            return q_mat, tost_q_mat
        else:
            return q_mat
        
        
    def get_signal_correlations(self,ds_list=None):
        r_act = np.ones((self.n_neurons,self.n_neurons))*np.nan
        count = np.ones((self.n_neurons,self.n_neurons))
        
        if ds_list is None: ds_list = np.arange(len(self.ds_list))
        
        for i_ds in ds_list:
            r_act_ = np.corrcoef(self.sig[i_ds].data.T)
            # Translate it in the atlas reference frame
            for i in np.arange(r_act_.shape[0]):
                ai = self.atlas_i[i_ds][i]
                if ai<0: continue
                for j in np.arange(r_act_.shape[1]):
                    aj = self.atlas_i[i_ds][j]
                    if aj<0: continue
                    if np.isnan(r_act[ai,aj]):
                        r_act[ai,aj] = r_act_[i,j]
                    else:
                        r_act[ai,aj] += r_act_[i,j]
                    count[ai,aj] += 1
        
        r_act[count!=0] = r_act[count!=0]/count[count!=0]
        
        return r_act
        
    def get_responses_correlations(self,occ2,interval,ds_list=None):
        if ds_list is None: ds_list = np.arange(len(self.ds_list))
        
        n_neu = len(occ2)
        r = np.ones((n_neu,n_neu))*np.nan
        count = np.zeros((n_neu,n_neu))
        
        for ds in ds_list:
            dt0 = int(interval[0]/self.fconn[ds].Dt)
            dt1 = int(interval[1]/self.fconn[ds].Dt)
            for ie in np.arange(self.fconn[ds].n_stim):
                if self.fconn[ds].stim_neurons[ie]>=0:
                    i0 = self.fconn[ds].i0s[ie]
                    i1 = self.fconn[ds].i1s[ie]
                    shift_vol = self.fconn[ds].shift_vols[ie]
                    
                    y = self.sig[ds].get_segment(
                                i0,i1,baseline=False,
                                normalize="none")[shift_vol-dt0:shift_vol+dt1]
                    
                    r_ = np.corrcoef(y.T)
                    for i in np.arange(r_.shape[0]):
                        ai = self.atlas_i[ds][i]
                        if ai<0: continue
                        for j in np.arange(r_.shape[1]):
                            aj = self.atlas_i[ds][j]
                            if aj<0: continue
                            if np.isnan(r[ai,aj]):
                                r[ai,aj] = r_[i,j]
                            else:
                                r[ai,aj] += r_[i,j]
                                
                            count[ai,aj] += 1
        r[count!=0] /= count[count!=0]
        
        return r
        
    def get__correlations_from_kernels_old(self,occ2,js=None):
        '''Compute, starting from the kernels, the correlations that would be 
        observed if neurons js were stimulated.
        
        Parameters
        ----------
        occ2:
            Precomputed occ2.
        js: array_like of int (optional)
            Indices of the neurons to be stimulated. If None, all neurons are 
            stimulated. Default: None.
        
        Returns
        -------
        A: numpy.ndarray of floats
            Correlation matrix.
        '''
        
        A = np.zeros((self.n_neurons,self.n_neurons))
        time = np.linspace(0,60,120)
        dt = time[1]-time[0]
        ec = pp.ExponentialConvolution([0.2,0.1])
        stim = ec.eval(time)
        psi = np.zeros((self.n_neurons,len(time)))
        
        if js is None: js = np.arange(self.n_neurons)
        
        for j in js:
            for i in np.arange(self.n_neurons):
                k = []
                for o in occ2[i,j]:
                    ds = o["ds"]
                    ie = o["stim"]
                    i_ds = o["resp_neu_i"]
                    
                    ec = self.fconn[ds].get_kernel_ec(ie,i_ds)
                    if ec is None: continue
                    k.append(ec.eval(time))
                if len(k)>0:
                    k = np.average(k,axis=0)
                    psi[i] += k#pp.convolution(k,stim,dt,8)
        
        
        A = np.corrcoef(psi)
        
        return A
        
    def get__correlation_from_kernels_full_of_old_stuff(self,occ2,occ3,q,js=None,time=None):
        '''Compute, starting from the kernels, the correlations that would be 
        observed if neurons js were stimulated.
        
        Parameters
        ----------
        occ2:
            Precomputed occ2.
        js: array_like of int (optional)
            Indices of the neurons to be stimulated. If None, all neurons are 
            stimulated. Default: None.
        
        Returns
        -------
        A: numpy.ndarray of floats
            Correlation matrix.
        '''
        
        A = np.zeros((self.n_neurons,self.n_neurons))
        if time is None:
            time = np.linspace(0,60,120)
        dt = time[1]-time[0]
        ec = pp.ExponentialConvolution([0.2,0.1])
        stim = ec.eval(time)
        psi = np.zeros((self.n_neurons,len(time)))
        count = np.zeros((self.n_neurons,self.n_neurons))
        flat_kernel = np.zeros_like(time)
        
        '''xdff,xsd,cdf = self.get_ctrl_cdf()
        
        n_ds = len(self.ds_list)
        sder = np.empty(n_ds,dtype=object)
        for a in sder: a = []
        for ds in np.arange(n_ds):
            sderker = savgol_coeffs(13, 2, deriv=2, delta=self.fconn[ds].Dt)
            sder_ = np.zeros_like(self.sig[ds].data)            
            for k in np.arange(self.sig[ds].data.shape[1]):
                sder_[:,k] = np.convolve(sderker,self.sig[ds].data[:,k],
                                         mode="same")
            sder[ds] = sder_'''
        
        if js is None: js = np.arange(self.n_neurons)
        
        for aj in js:
            psi[:] = np.nan
            psi[aj] = stim
            A_ = np.zeros_like(A)
            k = []
            w = []
            for ai in np.arange(self.n_neurons):
                k = []
                w = []
                for o in occ2[ai,aj]:
                    ds = o["ds"]
                    ie = o["stim"]
                    i_ds = o["resp_neu_i"]
                    j_ds = self.fconn[ds].stim_neurons[ie]
                    '''lbl_conf_i = self.labels_confidences[ds][i_ds]
                    lbl_conf_j = self.labels_confidences[ds][j_ds]
                    if lbl_conf_i>0 and lbl_conf_j>0:
                        lbl_conf = lbl_conf_i*lbl_conf_j
                    else:
                        continue'''
                    
                    ec = self.fconn[ds].get_kernel_ec(ie,i_ds)
                    stim_ec = self.fconn[ds].get_unc_fit_ec(ie,j_ds)
                    if ec is not None and stim_ec is not None: 
                        stim_y = stim_ec.eval(time)
                        stim_scale = np.max(np.abs(stim_y))
                        
                        ec.drop_saturation_branches()
                        k_ = ec.eval(time)*stim_scale
                        if np.any(np.isnan(k_)): continue
                        
                        k.append(k_)
                        w.append(1.0)
                    else:
                        k_ = flat_kernel
                        k.append(k_)
                        w.append(1.0)
                        
                    '''
                    i0 = self.fconn[ds].i0s[ie]
                    i1 = self.fconn[ds].i1s[ie]
                    shift_vol = self.fconn[ds].shift_vols[ie]
                    # build weights with the 1-p of the autoresponse of j
                    # (i.e. kernels fitted from small inputs should weight
                    # less)
                    
                    activity = self.sig[ds].get_segment(
                                        i0,i1,shift_vol,
                                        normalize="")[:,np.array([j_ds,i_ds])]
                    baseline = np.average(activity[:shift_vol],axis=0)
                    #pre = baseline
                    pre = self.sig[ds].get_loc_std(activity[:shift_vol],8)
                    act = np.average(activity[shift_vol:]-baseline,axis=0)/pre
                                        
                    sd_ = sder[ds][i0+shift_vol-5:i0+shift_vol+11,np.array([j_ds,i_ds])]
                    sd_ = np.sum(sd_,axis=0)
                    
                    dff__j,_,sd__j = self.get_significance_features(
                                                self.sig[ds],j_ds,i0,i1,shift_vol,
                                                self.fconn[ds].Dt,nan_th=0.3)
                    dff__i,_,sd__i = self.get_significance_features(
                                                self.sig[ds],i_ds,i0,i1,shift_vol,
                                                self.fconn[ds].Dt,nan_th=0.3)
                    if dff__j is not None and dff__i is not None:
                        act = [dff__j,dff__i]
                        sd_ = [sd__j,sd__i]
                        p_a_ = self.get_individual_p(act,xdff,cdf)
                        p_b_ = self.get_individual_p(sd_,xsd,cdf)
                        
                        p_a_[p_a_==0] = 1e-10
                        p_b_[p_b_==0] = 1e-10
                                       
                        _,p_j = combine_pvalues([p_a_[0],p_b_[0]],method="fisher")
                        _,p_i = combine_pvalues([p_a_[1],p_b_[1]],method="fisher")
                                                      
                        w_ = (1-p_i)*(1-p_j)
                        
                        k.append(k_)
                        w.append(w_)'''
                if len(k)>0:
                    k = np.average(np.array(k),axis=0,weights=w)
                    psi[ai] = pp.convolution(k,stim,dt,8)
                    #count[ai,aj] += 1
                    #count[aj,ai] += 1
            
            #sel = ~np.all(np.isnan(psi),axis=1)
            #A_[sel][:,sel] = np.corrcoef(psi[sel])
            A_ = np.corrcoef(psi)
            #A_ *= (1-p[:,j][:,None])*(1-p[:,j][None,:])
            A_ *= (1-q[:,aj][:,None])*(1-q[:,aj][None,:])
            if np.sum(~np.isnan(A_))>0:
                A[~np.isnan(A_)] = A[~np.isnan(A_)] + A_[~np.isnan(A_)]
        
        #A[count!=0] /= count[count!=0]
        A[occ3==0] = np.nan
            
        return A
        
    def get_correlation_from_kernels(self,occ2,occ3,q,js=None,time=None,
                                     shuffle_sorter=None):
        '''Compute, starting from the kernels, the correlations that would be 
        observed if neurons js were stimulated.
        
        Parameters
        ----------
        occ2:
            Precomputed occ2.
        js: array_like of int (optional)
            Indices of the neurons to be stimulated. If None, all neurons are 
            stimulated. Default: None.
        
        Returns
        -------
        A: numpy.ndarray of floats
            Correlation matrix.
        '''
        
        A = np.zeros((self.n_neurons,self.n_neurons))
        if time is None:
            time = np.linspace(0,60,120)
        dt = time[1]-time[0]
        ec = pp.ExponentialConvolution([0.2,0.1])
        stim = ec.eval(time)
        psi = np.zeros((self.n_neurons,len(time)))
        count = np.zeros((self.n_neurons,self.n_neurons))
        flat_kernel = np.zeros_like(time)
        
        if js is None: js = np.arange(self.n_neurons)
        
        for aj in js:
            psi[:] = np.nan
            psi[aj] = stim
            A_ = np.zeros_like(A)
            k = []
            w = []
            for ai in np.arange(self.n_neurons):
                k = []
                w = []
                for o in occ2[ai,aj]:
                    ds = o["ds"]
                    ie = o["stim"]
                    i_ds = o["resp_neu_i"]
                    j_ds = self.fconn[ds].stim_neurons[ie]
                    
                    ec = self.fconn[ds].get_kernel_ec(ie,i_ds)
                    stim_ec = self.fconn[ds].get_unc_fit_ec(ie,j_ds)
                    if ec is not None and stim_ec is not None: 
                        #stim_y = stim_ec.eval(time)
                        #stim_scale = np.max(np.abs(stim_y))
                        
                        ec.drop_saturation_branches()
                        k_ = ec.eval(time)#*stim_scale
                        if np.any(np.isnan(k_)): continue
                        
                        k.append(k_)
                        w.append(1.0)
                    else:
                        k_ = flat_kernel
                        k.append(k_)
                        w.append(1.0)
                        
                if len(k)>0:
                    k = np.average(np.array(k),axis=0,weights=w)
                    psi[ai] = pp.convolution(k,stim,dt,8)
                    #count[ai,aj] += 1
                    #count[aj,ai] += 1
            
            A_ = np.corrcoef(psi)
            #A_ *= (1-p[:,j][:,None])*(1-p[:,j][None,:])
            #A_ *= (1-q[:,aj][:,None])*(1-q[:,aj][None,:])
            if np.sum(~np.isnan(A_))>0:
                A[~np.isnan(A_)] = A[~np.isnan(A_)] + A_[~np.isnan(A_)]
                for ii in np.arange(A_.shape[0]):
                    for jj in np.arange(A_.shape[0]): #A.shape
                        if not np.isnan(A_[ii,jj]):  #A[]
                            count[ii,jj] += 1
        
        A[count!=0] /= count[count!=0]
        A[occ3==0] = np.nan
            
        return A
        
    def get_kernels_map(self,occ2,occ3,js=None,time=None,filtered=True,
                        drop_saturation_branches=False,
                        include_flat_kernels=False):
        '''Return the array of the average kernels.
        '''
        
        if time is None:
            time = np.linspace(0,60,120)
        dt = time[1]-time[0]
        
        kernels = np.zeros((self.n_neurons,self.n_neurons,len(time)))
        flat_kernel = np.zeros_like(time)
        
        if filtered:
            ec = pp.ExponentialConvolution([0.2,0.1])
            stim = ec.eval(time)
            
        if js is None: js = np.arange(self.n_neurons)
        
        for aj in js:
            for ai in np.arange(self.n_neurons):
                k = []
                for o in occ2[ai,aj]:
                    ds = o["ds"]
                    ie = o["stim"]
                    i_ds = o["resp_neu_i"]
                    j_ds = self.fconn[ds].stim_neurons[ie]
                    
                    ec = self.fconn[ds].get_kernel_ec(ie,i_ds)
                    stim_ec = self.fconn[ds].get_unc_fit_ec(ie,j_ds)
                    if ec is not None and stim_ec is not None:
                        stim_y = stim_ec.eval(time)
                        stim_scale = np.max(np.abs(stim_y))
                        
                        if drop_saturation_branches:
                            ec = ec.drop_saturation_branches()
                        k_ = ec.eval(time)#*stim_scale
                        if np.any(np.isnan(k_)): continue
                        
                        k.append(k_)
                    elif include_flat_kernels:
                        k_ = flat_kernel
                        k.append(k_)
                        
                if len(k)>0:
                    kernels[ai,aj] = np.average(np.array(k),axis=0)
                    if filtered:
                        kernels[ai,aj] = pp.convolution(kernels[ai,aj],stim,dt,8)
        
        kernels[occ3==0] = np.nan
                        
        return kernels
        
    def get_kernels_map_ec(self,occ2,occ3,js=None,drop_saturation_branches=False,
                        include_flat_kernels=False):
        '''Return the array of the average kernels. None means that there is no
        data.
        '''
        
        kernels = np.empty((self.n_neurons,self.n_neurons),dtype=object)
        flat_kernel = pp.ExponentialConvolution([1.],0.0)
        
        if js is None: js = np.arange(self.n_neurons)
        
        for aj in js:
            for ai in np.arange(self.n_neurons):
                ec_avg = flat_kernel.copy()
                ec_count = 0
                for o in occ2[ai,aj]:
                    ds = o["ds"]
                    ie = o["stim"]
                    i_ds = o["resp_neu_i"]
                    j_ds = self.fconn[ds].stim_neurons[ie]
                    
                    ec = self.fconn[ds].get_kernel_ec(ie,i_ds)
                    if ec is not None:
                        if drop_saturation_branches:
                            ec = ec.drop_saturation_branches()
                        
                        if ec_avg is None:
                            ec_avg = ec
                        else:
                            ec_avg = ec_avg.add_simple(ec_avg,ec)    
                        ec_count += 1                    

                    elif include_flat_kernels:
                        ec_count += 1 
                        
                if ec_count>0 and ec_avg is not None:
                    ec_avg.multiply_scalar_inplace(1./ec_count)
                
                kernels[ai,aj] = ec_avg
        
        kernels[occ3==0] = None
                        
        return kernels
        
    def get_correlation_from_kernels_map(self,kernels,occ3,js=None,
                                         set_unknown_to_zero=False):
        '''Get the kernels-generated correlations from the precomputed kernel
        map.'''
        
        if js is None: 
            assert kernels.shape[0] == self.n_neurons
            js = np.arange(self.n_neurons)
        
        #A = np.zeros((len(js),self.n_neurons,self.n_neurons))
        #count = np.zeros((self.n_neurons,self.n_neurons))
        A = np.zeros((len(js),kernels.shape[0],kernels.shape[1]))
        
        for i_aj in np.arange(len(js)):
            aj = js[i_aj]
            A_ = np.corrcoef(kernels[:,aj])
            A[i_aj] = A_
            
        if set_unknown_to_zero:
            #fil1 = ~np.all(np.isnan(A),axis=0)
            #A__ = A[...,fil1]
            #A[...,fil1][np.isnan(A__)] = 0.0
            A[np.isnan(A)] = 0.0
        A = np.nanmean(A,axis=0)
        A[occ3==0] = np.nan
        
        return A
        
    def get_correlation_from_kernels_full_trace(self,occ2,occ3,q,js=None,time=None):
        '''Compute, starting from the kernels, the correlations that would be 
        observed if neurons js were stimulated.
        
        Parameters
        ----------
        occ2:
            Precomputed occ2.
        js: array_like of int (optional)
            Indices of the neurons to be stimulated. If None, all neurons are 
            stimulated. Default: None.
        
        Returns
        -------
        A: numpy.ndarray of floats
            Correlation matrix.
        '''
        
        A = np.zeros((self.n_neurons,self.n_neurons))
        if time is None:
            time = np.linspace(0,60,120)
        dt = time[1]-time[0]
        ec = pp.ExponentialConvolution([0.2,0.1])
        stim = ec.eval(time)
        psi = np.zeros((self.n_neurons,self.n_neurons,len(time)))
        count = np.zeros((self.n_neurons,self.n_neurons))
        flat_kernel = np.zeros_like(time)
        
        if js is None: js = np.arange(self.n_neurons)
        
        for aj in js:
            psi[aj,aj] = stim
            A_ = np.zeros_like(A)
            k = []
            w = []
            for ai in np.arange(self.n_neurons):
                k = []
                w = []
                for o in occ2[ai,aj]:
                    ds = o["ds"]
                    ie = o["stim"]
                    i_ds = o["resp_neu_i"]
                    j_ds = self.fconn[ds].stim_neurons[ie]
                    
                    ec = self.fconn[ds].get_kernel_ec(ie,i_ds)
                    stim_ec = self.fconn[ds].get_unc_fit_ec(ie,j_ds)
                    if ec is not None and stim_ec is not None: 
                        stim_y = stim_ec.eval(time)
                        stim_scale = np.max(np.abs(stim_y))
                        
                        ec.drop_saturation_branches()
                        k_ = ec.eval(time)*stim_scale
                        if np.any(np.isnan(k_)): continue
                        
                        k.append(k_)
                        w.append(1.0)
                    else:
                        k_ = flat_kernel
                        k.append(k_)
                        w.append(1.0)
                        
                if len(k)>0:
                    k = np.average(np.array(k),axis=0,weights=w)
                    psi[ai,aj] = pp.convolution(k,stim,dt,8)
            
        A = np.corrcoef(psi.reshape((self.n_neurons,self.n_neurons*len(time))))
            
        return A
    
    #########################
    #########################
    #########################
    # ANATOMICAL CONNECTOME
    #########################
    #########################
    #########################
    
    
    def load_aconnectome_from_file(self,chem_th=3,gap_th=2,exclude_white=False,
                                   average=False):
        self.aconn_chem, self.aconn_gap = \
                    self.get_aconnectome_from_file(chem_th,gap_th,exclude_white,
                                                   average=average)
        
    def get_aconnectome_from_file(self,chem_th=3,gap_th=2,exclude_white=False,
                                  average=False):
        '''Load the anatomical connectome data from all the sources listed in 
        the class.
        
        Returns
        -------
        chem: numpy.ndarray
            chem[i,j] is the count of chemical synapses from j to i, averaged
            across the sources.
        gap: numpy.ndarray
            gap[i,j] is the count of gap junctions from j to i, averaged
            across the sources.
        '''
        chem = np.zeros((self.n_neurons, self.n_neurons))
        gap = np.zeros((self.n_neurons, self.n_neurons))
        
        sources_used = 0
        for source in self.aconn_sources:
            if source["type"]=="white" and not exclude_white:
                c, g = self._get_aconnectome_white(
                                        self.module_folder+source["fname"],
                                        self.module_folder+source["ids_fname"])
            elif source["type"] in ["whiteL4","whiteA"] and not exclude_white:
                c, g = self._get_aconnectome_witvliet(
                                        self.module_folder+source["fname"])
            elif source["type"]=="witvliet":
                c, g = self._get_aconnectome_witvliet(
                                        self.module_folder+source["fname"])
            else:
                continue
            
            chem += c
            gap += g
            sources_used += 1
        
        if average:    
            chem /= sources_used
            gap /= sources_used
        
        chem[chem<=chem_th] = 0
        gap[gap<=gap_th] = 0
            
        return chem, gap
        
    def _get_aconnectome_white(self,fname,ids_fname):
        '''Load the anatomical connectome data from the file format for the
        White dataset.
        
        Returns
        -------
        chem: numpy.ndarray
            chem[i,j] is the count of chemical synapses from j to i.
        gap: numpy.ndarray
            gap[i,j] is the count of gap junctions from j to i.
        '''
        chem = np.zeros((self.n_neurons,self.n_neurons))
        gap = np.zeros((self.n_neurons,self.n_neurons))
        
        # Load the connectome matrices
        f = open(fname,"r")
        conn = json.load(f)
        f.close()
        
        # Load the associated ids
        white_ids = []
        f = open(ids_fname,'r')
        for l in f.readlines():
            neu_name = l.split("\t")[1]
            if neu_name[-1:] == "\n": neu_name = neu_name[:-1]
            white_ids.append(neu_name)
        f.close()
        
        a_is = self.ids_to_i(white_ids)
        
        for i_w in np.arange(self.n_neurons):
            a_i = a_is[i_w]
            for j_w in np.arange(self.n_neurons):
                a_j = a_is[j_w]
                # The matrices have to be transposed, but they're lists
                # so switch i and j
                chem[a_i,a_j] += conn["chemical"][j_w][i_w]
                gap[a_i,a_j] += conn["electrical"][j_w][i_w]
                if conn["electrical"][i_w][j_w] == 0:
                    gap[a_j,a_i] += conn["electrical"][j_w][i_w]
                #chem[a_i,a_j] = conn["chemical"][i_w][j_w]
                #gap[a_i,a_j] = conn["electrical"][i_w][j_w]
                
        return chem, gap
        
    def _get_aconnectome_witvliet(self,fname):
        '''Load the anatomical connectome data from the file format for the
        Witvliet dataset, or datasets stored by Witvliet in Nemanode.
        
        Returns
        -------
        chem: numpy.ndarray
            chem[i,j] is the count of chemical synapses from j to i.
        gap: numpy.ndarray
            gap[i,j] is the count of gap junctions from j to i.
        '''
        chem = np.zeros((self.n_neurons,self.n_neurons))
        gap = np.zeros((self.n_neurons,self.n_neurons))
        
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        
        for l in lines[1:]:
            sl = l.split("\t")
            id_from = sl[0]
            id_to = sl[1]
            conn_type = sl[2]
            conn_n = int(sl[3])
            
            if conn_n!=0:
                ai_from = self.ids_to_i(id_from)
                ai_to = self.ids_to_i(id_to)
                
                if conn_type == "chemical":
                    chem[ai_to,ai_from] += conn_n
                elif conn_type == "electrical":
                    gap[ai_to,ai_from] += conn_n
                    #gap[ai_from,ai_to] += conn_n
                    
        for i in np.arange(self.n_neurons):
            for j in np.arange(self.n_neurons):
                gap[i,j] = max(gap[i,j],gap[j,i])
                
        return chem, gap
        
    def get_effective_aconn(self,s=1):
        '''s is a modulation of the strenght'''
        g = s*(self.aconn_chem+self.aconn_gap)
        diag_i = np.diag_indices(self.n_neurons)
        g[diag_i] = 0.0
        
        G = np.zeros_like(g)
        
        for j in np.arange(self.n_neurons):
            b = np.copy(g[:,j])
            a = np.copy(-g)
            a[:,j] = 0.0
            a[j,j] = -g[j,j]
            for i in np.arange(self.n_neurons): a[i,i] += 1.
            
            G[:,j] = np.linalg.solve(a,b)
            
        return G
        
    def get_effective_aconn2(self,max_hops=None):
        '''Returns the boolean anatomical connectome either at convergence or
        with a maximum number of hops.
        
        Parameters
        ----------
        max_hops: int (optional)
            Maximum number of hops to allow for the effective connectivity. If
            None, the effective connectivity is returned at convergence. The
            boolean single-hop anatomical connectome is returned with max_hops
            equal to 1. Default: None.
            
        Returns
        -------
        c: 2D numpy.ndarray of int
            Effective connectome. Values are either 0, if the neurons are in
            disjoint sets, or 1, if the neuron are connected.        
        '''
        c = (self.aconn_chem+self.aconn_gap)
        cl = np.min(c[c>0])
        c = np.clip(c,0,cl)/cl
        
        # The starting c is already with 1 hop (from 0 would be additional hops)
        p = 1
        if max_hops is None: max_hops = 10000
        while True:
            if p==max_hops: break
            old_c = c.copy()
            # c_ij = c_ik c_kj
            for i in np.arange(len(c)):
                for j in np.arange(len(c)):
                    c[i,j] = old_c[i,j]+np.sum(old_c[i,:]*old_c[:,j])
            c = np.clip(c,0,1)
            if np.all(c==old_c): break
            p +=1
            
        return c
        
    def get_effective_aconn3(self,max_hops=None,gain_1='average',c=None):
        '''Returns the anatomical connectome either at convergence or
        with a maximum number of hops, by setting the gain equal to 1 for 
        a given number of contacts.
        
        Parameters
        ----------
        max_hops: int (optional)
            Maximum number of hops to allow for the effective connectivity. If
            None, the effective connectivity is returned at convergence. The
            boolean single-hop anatomical connectome is returned with max_hops
            equal to 1. Default: None.
            
        gain_1: str or int (optional)
            Where to set the gain equal to 1 in terms of number of contacts.
            If it is set to 'average', the average number of contacts 
            corresponds to gain=1. Default: average.
            
        Returns
        -------
        c: 2D numpy.ndarray of int
            Effective connectome.   
        '''
        if c is None:
            c = (self.aconn_chem+self.aconn_gap)
        
        if gain_1=='average':
            # Set the average number of connections to gain 1
            gain_1 = np.average(c)
            
        c /= gain_1
        orig_max = np.max(c)
            
        # The starting c is already with 1 hop (from 0 would be additional hops)
        p = 1
        if max_hops is None: max_hops = 10000
        while True:
            if p==max_hops: break
            old_c = c.copy()
            # c_ij = c_ik c_kj
            for i in np.arange(len(c)):
                for j in np.arange(len(c)):
                    c[i,j] = old_c[i,j]+np.sum(np.delete(old_c[i,:]*old_c[:,j],(i,j)))
            #c *= orig_max/np.max(c)
            if np.allclose(c,old_c): 
                print("Aconnectome converged in",p); break
            p +=1
            
        return c
        
    def get_effective_aconn4(self,max_hops=None,gain_1='average',c=None,
                             return_all=False): 
        '''Returns the anatomical connectome either at convergence or
        with a maximum number of hops, by setting the gain equal to 1 for 
        a given number of contacts.
        
        Parameters
        ----------
        max_hops: int (optional)
            Maximum number of hops to allow for the effective connectivity. If
            None, the effective connectivity is returned at convergence. The
            boolean single-hop anatomical connectome is returned with max_hops
            equal to 1. Default: None.
            
        gain_1: str or int (optional)
            Where to set the gain equal to 1 in terms of number of contacts.
            If it is set to 'average', the average number of contacts 
            corresponds to gain=1. Default: average.
            
        Returns
        -------
        c: 2D numpy.ndarray of int
            Effective connectome.   
        '''
        if c is None:
            c = (self.aconn_chem+self.aconn_gap)
        
        if gain_1=='average':
            # Set the average number of connections to gain 1
            gain_1 = np.average(c)
            
        c /= gain_1
        orig_max = np.max(c)
            
        # The starting c already contains the 1 hop
        p = 2
        if max_hops is None: max_hops = 10000
        old_c = np.copy(c)
        while True:
            if p==max_hops: break
            c += np.linalg.matrix_power(c,p)
            
            if np.allclose(c,old_c): 
                print("Aconnectome converged in",p); break
            p +=1
            old_c = np.copy(c)
        
        if return_all:
            return c,p
        else:
            return c
        
        
    @staticmethod
    def _converged_matrix_power(A,n_max=100,eps=1e-2,return_all=False):
        A0 = A
        A_old = A
        n = 0
        e = 2*eps
        while True and n<n_max and e>eps:
            A = np.dot(A,A0)
            e = np.sum(np.absolute(A_old-A))/np.sum(A_old)
            A_old = A
            n += 1
        
        if return_all:
            return A,n
        else:
            return A
            
    @staticmethod
    def corr_from_eff_causal_conn(A):
        '''B = 0.5*(A+A.T)
        C = np.zeros_like(B)
        count = np.zeros_like(B)
        
        n = A.shape[0]
        ns = np.arange(n)
        
        for i in np.arange(n):
            for k in np.arange(n):
                if k==i: continue
                for j in np.arange(n):
                    if j==i or j==k: continue
                    if np.isfinite(A[i,j]) and np.isfinite(A[k,j]):
                        C[i,k] += A[i,j]*A[k,j]
                        count[i,k] += 1
        B[count!=0] += C[count!=0]#/count[count!=0]'''
        
        #return B 
        return 0.5*(A+A.T)+np.dot(A,A.T)
    
    @staticmethod
    def symmetrize_nan_preserving(A):
        nanmask = np.isnan(A)*np.isnan(A.T)
        A_ = 0.5*(np.nansum([A,A.T],axis=0))
        A_[nanmask] = np.nan
        
        return A_
            
    
    @staticmethod    
    def _get_next_hop_aconn(c):
        '''Compute the boolean map of strictly-2-hops connections given an input 
        connectome. To be used recursively in the calculation of the 
        strictly-n-hops connections.
        
        Parameters
        ----------
        c: numpy.ndarray of bool
            Input boolean connectome.
        
        Returns
        -------
        c_new: numpy.ndarray of bool
            Output strictly-2-hop boolean connectome.
        
        '''
        n = c.shape[0]
        
        c_new = np.zeros_like(c,dtype=bool)
        for j in np.arange(n):
            for i in np.arange(n):
                c_new[i,j] = not c[i,j] and np.any(c[i,:]*c[:,j])
                        
        return c_new
        
    def get_n_hops_aconn(self,n_hops):
        '''Return the anatomical connectome at n_hops from the original
        connectome.
        
        Parameters
        ----------
        n_hops: int
            Number of hops.
        
        Returns
        -------
        c_new: numpy.ndarray of bool
            Connectome n_hops away from the original one.
        '''
        
        c = self.get_boolean_aconn()
        if n_hops == 1: return c
        
        c_prev_hops = np.copy(c)
        for ih in np.arange(n_hops-1):
            # Get the connectome exactly 1-hop away from the current 
            # effective connectome.
            c_new = self._get_next_hop_aconn(c_prev_hops)
            # Update the current effective connectome by including all previous
            # connections and the new connections that you just found (any 
            # connection that has already been found corresponds to a connection
            # with less than the desired, exact n_hops).
            c_prev_hops = c_prev_hops+c_new
            
        return c_new
        
    def get_boolean_aconn(self):
        '''Returns the boolean anatomical connectome.
        
        Returns
        -------
        c: numpy.ndarray of bool
            Boolean anatomical connectome.
        '''
        
        c = (self.aconn_chem+self.aconn_gap) != 0
        return c
        
            
        
    def get_anatomical_paths(self,*args,**kwargs):
        '''Find all the paths between two neurons within a maximum numer of 
        hops.See Funatlas._get_paths() for the function arguments and returns. 
        In the  function arguments, skip the connectome matrix conn, which, 
        here, is set directly as the anatomical connectome.
        '''
        
        conn = self.aconn_chem + self.aconn_gap
        return self._get_paths(conn, *args, **kwargs)
                  
    def _get_paths(self,conn,i,j,max_n_hops=1,
                             return_ids=False,exclude_self_loops=True):
        '''Given a connectome, find all the paths between two neurons within a 
        maximum numer of hops.
        
        Parameters
        ----------
        conn: numpy.ndarray
            Connectome matrix.
        i: int or str
            Atlas-index or ID of the downstream neuron.
        j: int or str
            Atlas-index or ID of the upstream neuron.
        max_n_hops: int (optional)
            Maximum number of hops in which to perform the search. Default: 1.
        return_ids: bool (optional)
            Return also the paths with the IDs of the neurons, instead of their
            atlas-indices. Default: False.
        exclude_self_loops: bool (optional)
            Whether to exclude recurrent paths looping on the downstream
            neuron i.
            
        Returns
        -------
        paths: list of lists of integers
            paths[p][q] is the atlas index of q-th neuron on the p-th path. The 
            paths are ordered downstream-upstream.
        paths: list of lists of strings
            paths[p][q] is the index of q-th neuron on the p-th path. The 
            paths are ordered downstream-upstream. Returned only if return_ids
            is True.    
        '''
                             
        if type(i)==str: i = self.ids_to_i(i)
        if type(j)==str: j = self.ids_to_i(j)
        # You could add a check with a scalar Dyson equation to see if there
        # is a path at all. You can use the effective anatomical connectivity
        # at all steps to help speed up the process.
        paths_final = []
        
        paths_all = []
        # Populate paths_all with all the 1-hop connections that send signals
        # into i.
        for q in np.arange(self.n_neurons):
            if conn[i,q] != 0:
                paths_all.append([i,q])

        for h in np.arange(max_n_hops):
            paths_all_new = []
            for p in paths_all:
                if p[-1] == j: 
                    # Arrived at j
                    paths_final.append(p)
                elif h!=max_n_hops-1:
                    # Iterate over all the connections and add a hop
                    for q in np.arange(self.n_neurons):
                        if conn[p[-1],q]!=0 \
                            and not (exclude_self_loops and q==i):
                            new_p = p.copy()
                            new_p.append(q)
                            paths_all_new.append(new_p)
            paths_all = paths_all_new.copy()
        
        for p in paths_all:
            if p[-1] == j: 
                # Arrived at j
                paths_final.append(p)
                
        if return_ids:
            paths_final_ids = paths_final.copy()
            for i_p in np.arange(len(paths_final)):
                for q in np.arange(len(paths_final[i_p])):
                    paths_final_ids[i_p][q] = self.neuron_ids[paths_final_ids[i_p][q]]
            
            return paths_final, paths_final_ids
        else:
            return paths_final
            
    
    @staticmethod
    def _have_common_1_hop_upstream(c):
        
        n = len(c)
        c_new = np.zeros_like(c)
        
        for i in np.arange(n):
            for k in np.arange(n):
                # c_new[i,k] is True if there is at least one c[i,:] that is also
                # in c[k,:].
                c_new[i,k] = np.sum(c[i,:]*c[k,:])!=0
                
        return c_new
        
    def have_common_n_hop_upstream(self,n=1):
        if n!=1: raise ValueError("Implemented for n=1 only.")

        c = (self.aconn_chem+self.aconn_gap) != 0
        
        c_new = self._have_common_1_hop_upstream(c)
        
        return c_new
        
    def shuffle_aconnectome(self,shuffling_sorter=None):
        if shuffling_sorter is None:
            shuffling_sorter = self.get_shuffling_sorter()
            
        self.aconn_chem = self.shuffle_array(self.aconn_chem,shuffling_sorter)
        self.aconn_gap = self.shuffle_array(self.aconn_gap,shuffling_sorter)
    
    
    def load_innexin_expression_from_file(self):
        if not self.merge_bilateral:
            print("inx expression level available only with merge_bilateral")
            return None
        else:
            f = open(self.module_folder+self.fname_innexins,"r")
            lines = f.readlines()
            f.close()
            
            ids = lines[0][:-1].split(",")[4:]
            
            exp_levels = np.zeros((self.n_neurons,len(lines)-1))*np.nan
            genes = []
            
            for il in np.arange(len(lines)-1):
                l = lines[il+1]
                a=l.split(",")
                genes.append(a[1])
                
                for iid in np.arange(len(ids)):
                    names = self.cengen_ids_conversion([ids[iid]])[0]
                    '''if ids[iid]=="IL1":
                        names = ["IL1_","IL1D_","IL1V_"]
                    elif ids[iid]=="IL2_DV":
                        names = ["IL2D_","IL2V_"]
                    elif ids[iid]=="IL2_LR":
                        names = ["IL2_"]
                    elif ids[iid]=="CEP":
                        names = ["CEPD_","CEPV_"]
                    elif ids[iid]=="OLQ":
                        names = ["OLQD_","OLQV_"]
                    elif ids[iid]=="RMD_DV":
                        names = ["RMDD_","RMDV_"]
                    elif ids[iid]=="RMD_LR":
                        names = ["RMD_"]
                    elif ids[iid]=="RME_DV":
                        names = ["RMED","RMEV"]
                    elif ids[iid]=="RME_LR":
                        names = ["RME_"]
                    elif ids[iid]=="SMD":
                        names = ["SMDD_","SMDV_"]
                    elif ids[iid]=="URY":
                        names = ["URYD_","URYV_"]
                    elif ids[iid]=="URA":
                        names = ["URAD_","URAV_"]
                    elif ids[iid]=="SAA":
                        names = ["SAAD_","SAAV_"]
                    elif ids[iid]=="SAB":
                        names = ["SABD_","SABV_"]
                    elif ids[iid]=="SIA":
                        names = ["SIAD_","SIAV_"]
                    elif ids[iid]=="SIB":
                        names = ["SIBD_","SIBV_"]
                    elif ids[iid]=="SMB":
                        names = ["SMBD_","SMBV_"]
                    else:
                        names = [ids[iid]]'''
                    
                    for name in names:
                        aid = self.ids_to_i(name)
                        if aid<0: continue
                        exp_levels[aid,il] = float(a[4+iid])
                    
            self.inx_exp_levels = exp_levels
            self.inx_genes = genes
                    
    def get_inx_exp_levels(self):
        unc7_i = self.inx_genes.index("unc-7")
        unc9_i = self.inx_genes.index("unc-9")

        unc7_9_to_others = np.zeros(self.n_neurons)
        
        for ai in np.arange(self.n_neurons):
            u79 = (self.inx_exp_levels[ai,unc7_i]+self.inx_exp_levels[ai,unc9_i])
            if np.sum(self.inx_exp_levels[ai])>0:
                unc7_9_to_others[ai] = u79/np.sum(self.inx_exp_levels[ai])
        
        return self.inx_exp_levels, self.inx_genes, unc7_9_to_others
        
    def get_inx_exp_levels2(self,inx):
        inx_i = self.inx_genes.index(inx)
        
        fr = np.zeros(self.n_neurons)
        for ai in np.arange(self.n_neurons):
            tot_inx = np.sum(self.inx_exp_levels[ai])
            if tot_inx>0:
                fr[ai] = self.inx_exp_levels[ai,inx_i]/tot_inx
                
        return fr
        
    def get_fractional_gap_inx_mutants(self,mode='min'):
        '''Returns the remaining fraction of gap junctions, estimated by the
        product of the respective non-unc-7/9 fraction of innexins in the two
        neurons.        
        '''
        fr = np.ones((self.n_neurons,self.n_neurons))
        
        _,_,u79to = self.get_inx_exp_levels()
        
        if mode=="mult":
            fr *= (1-u79to[:,None])
            fr *= (1-u79to[None,:])
        else:
            for i in np.arange(fr.shape[0]):
                for j in np.arange(fr.shape[1]):
                    fr[i,j] = min( 1-u79to[i] , 1-u79to[j] )
        
        return fr
        
    def load_neuropeptide_expression_from_file(self):
        if not self.merge_bilateral:
            print("neuropeptide expression level available only with merge_bilateral")
            return None
        else:
            f = open(self.module_folder+self.fname_neuropeptides,"r")
            lines = f.readlines()
            f.close()
            
            ids = lines[0][:-1].split(",")[4:]
            
            exp_levels = np.zeros((self.n_neurons,len(lines)-1))*np.nan
            genes = []
            
            for il in np.arange(len(lines)-1):
                l = lines[il+1]
                a=l.split(",")
                genes.append(a[1])
                
                for iid in np.arange(len(ids)):
                    names = self.cengen_ids_conversion([ids[iid]])[0]
                    
                    for name in names:
                        aid = self.ids_to_i(name)
                        if aid<0: continue
                        if np.isnan(exp_levels[aid,il]):
                            exp_levels[aid,il] = float(a[4+iid])
                        else:
                            exp_levels[aid,il] += float(a[4+iid])
                    
            self.npt_exp_levels = exp_levels
            self.npt_genes = genes
            
    def load_neuropeptide_receptor_expression_from_file(self):
        if not self.merge_bilateral:
            print("neuropeptide receptor expression level available only with merge_bilateral")
            return None
        else:
            f = open(self.module_folder+self.fname_neuropeptide_receptors,"r")
            lines = f.readlines()
            f.close()
            
            ids = lines[0][:-1].split(",")[4:]
            
            exp_levels = np.zeros((self.n_neurons,len(lines)-1))*np.nan
            genes = []
            
            for il in np.arange(len(lines)-1):
                l = lines[il+1]
                a=l.split(",")
                genes.append(a[1])
                
                for iid in np.arange(len(ids)):
                    names = self.cengen_ids_conversion([ids[iid]])[0]
                    
                    for name in names:
                        aid = self.ids_to_i(name)
                        if aid<0: continue
                        if np.isnan(exp_levels[aid,il]):
                            exp_levels[aid,il] = float(a[4+iid])
                        else:
                            exp_levels[aid,il] += float(a[4+iid])
                    
            self.nptr_exp_levels = exp_levels
            self.nptr_genes = genes
    
    @staticmethod      
    def _get_aconn_sign_genetic_prediction():
        #Downlaod the Excel workbook from the paper
        import shutil
        import tempfile
        import urllib.request
        url = 'https://doi.org/10.1371/journal.pcbi.1007974.s003'
        with urllib.request.urlopen(url) as response:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                shutil.copyfileobj(response, tmp_file) #store it in a temporary location

        import pandas as pd
        WS = pd.read_excel(tmp_file.name, sheet_name='5. Sign prediction')
        WS_np = np.array(WS)

        # Location of various data within the excel worksheet
        pre_syn_col = 0
        post_syn_col = 3
        pred_col = 16

        first_row = 2
        last_row = 3639

        pre = WS_np[first_row:last_row, pre_syn_col] #Presynaptic neuron
        post = WS_np[first_row:last_row, post_syn_col] #postsynaptic neuron
        sign = WS_np[first_row:last_row, pred_col] #Sign prediction
        # strip out the 0's from neuron names to match our formatting so that we get VB1 instead of VB01
        pre = pd.Series(pre).str.replace('0','').to_numpy()
        post = pd.Series(post).str.replace('0','').to_numpy()
        return pre, post, sign

    def get_aconn_sign_genetic_prediction(self):
        pre, post, pred = self._get_aconn_sign_genetic_prediction()
        sign = np.ones((self.n_neurons,self.n_neurons))
        for k in np.arange(len(pre)):
            if pred[k] == "-":
                aj,ai = self.ids_to_i([pre[k],post[k]])
                sign[ai,aj] = -1
        return sign
        
    
    ##########################
    ##########################
    ##########################
    # EXTRASYNAPTIC CONNECTOME
    ##########################
    ##########################
    ##########################
    
    def load_extrasynaptic_connectome_from_file(self, *args, **kwargs):
         esconn_ma, esconn_np = self.get_extrasynaptic_connectome_from_file(
                                                *args,**kwargs)
         self.esconn_ma = esconn_ma
         self.esconn_np = esconn_np
    
    def get_extrasynaptic_connectome_from_file(self, transmitter_types=None,
                                               *args,**kwargs):
        '''Load the boolean extrasynaptic connectome data from all the sources 
        listed in the class.
        
        Parameters
        ----------
        transmitter_types: str or list of str (optional)
            Types of transmitters to allow (monoamines, neuropeptides, ...).
            If None, all are selected. Default: None.
        args, kwargs
        
        Returns
        -------
        ma: numpy.ndarray of bool
            Monoamine extrasynaptic connectome.
        np: numpy.ndarray of bool
            Neuropeptide extrasynaptic connectome.
        
        '''
        esconn_ma = np.zeros((self.n_neurons, self.n_neurons),dtype=bool)
        esconn_np = np.zeros((self.n_neurons, self.n_neurons),dtype=bool)
        
        if transmitter_types is not None:
            if type(transmitter_types)==str:
                transmitter_types = [transmitter_types]
        
        sources_used = 0
        for source in self.esconn_sources:
            if transmitter_types is not None:
                if source["transmitter_type"] not in transmitter_types:
                    continue
            if source["type"]=="bentley":
                esc = self._get_esconnectome_bentley(
                            self.module_folder+source["fname"],
                            *args,**kwargs)
            else:
                continue
            
            if source["transmitter_type"] == "monoamines":
                esconn_ma = np.logical_or(esconn_ma,esc)
            elif source["transmitter_type"] == "neuropeptides":
                esconn_np = np.logical_or(esconn_np,esc)
            
            sources_used += 1
        
        return esconn_ma, esconn_np
    
    def _get_esconnectome_bentley(self,fname,transmitters=None,receptors=None):
        '''Returns the extrasynaptic connectome from Bentley et al. 2016 "The 
        multilayer connectome of Caenorhabditis elegans" PLOS Comp. Bio.
        
        Parameters
        ----------
        fname: str
            Name of the csv file.
        transmitter: str or list of str (optional)
            Requested trasmitters. If None, no restriction is applied. 
            Default: None.
        receptors: str or list of str (optional)
            Requested receptors. If None, no restriction is applied. 
            Default: None.
        
        Returns
        -------
        esc: numpy.ndarray of bool
            Extrasynaptic connectome given the requested transmitters and
            receptors.
            
        '''
        esc = np.zeros((self.n_neurons,self.n_neurons),dtype=bool)
        
        if transmitters is not None:
            if type(transmitters)==str:transmitters = [transmitters]
        if receptors is not None:
            if type(receptors)==str:receptors = [receptors]
                
        
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        
        for l in lines[1:]:
            sl = l.split(",")
            id_from = sl[0]
            id_to = sl[1]
            trans = sl[2]
            recept = sl[3]
            if recept[-1] == "\n": recept = recept[-1]
            
            # Skip the line if the transmitter/receptor are not the requested
            # ones.
            if transmitters is not None:
                if trans not in transmitters: continue
            if receptors is not None:
                if recept not in receptors: continue
            
            ai_from = self.ids_to_i(id_from)
            ai_to = self.ids_to_i(id_to)
            
            esc[ai_to,ai_from] = True
                
        return esc
    
    def have_common_n_hop_es_upstream(self,n=1):
        if n!=1: raise ValueError("Implemented for n=1 only.")

        c = self.get_esconn()
        
        c_new = self._have_common_1_hop_upstream(c)
        
        return c_new
    
    
    def get_esconn(self):
        print("get_esconn changed ^ to +")
        return self.esconn_ma+self.esconn_np
        
    def get_effective_esconn(self,maxit=100):
        escon = self.get_esconn()
        eescon = np.copy(escon)
        old_eescon = np.copy(escon)
        it = 0
        while True and it<maxit:
            for j in np.arange(escon.shape[0]):
                for i in np.arange(escon.shape[0]):
                    if escon[i,j]:
                        eescon[escon[:,i],j] = True
            if np.all(old_eescon==eescon): break
            old_eescon = eescon
            it+=1
        return eescon
        
    def get_extrasynaptic_paths(self, *args, **kwargs):
        '''Returns the extrasynaptic paths between neurons. See 
        Funatlas._get_paths() for the function arguments and returns. In the 
        function arguments, skip the connectome matrix conn, which, here,
        is set directly as the extrasynaptic connectome.
        '''
        
        conn = self.get_esconn()
        return self._get_paths(conn, *args, **kwargs)
        
    def shuffle_esconnectome(self,shuffling_sorter=None):
        if shuffling_sorter is None:
            shuffling_sorter = self.get_shuffling_sorter()
            
        self.esconn_ma = self.shuffle_array(self.esconn_ma,shuffling_sorter)
        self.esconn_np = self.shuffle_array(self.esconn_np,shuffling_sorter)
        
    def get_RID_downstream(self,average=True):
        #fnames = ["../../preliminary_scripts/external_data/GenesExpressing-npr-4-thrs2.csv",
        #  "../../preliminary_scripts/external_data/GenesExpressing-npr-11-thrs2.csv",
        #  "../../preliminary_scripts/external_data/GenesExpressing-pdfr-1-thrs2.csv",
        #  #"external_data/GenesExpressing-daf-2-thrs2.csv"
        #  ]
          
        fnames = [self.module_folder+"GenesExpressing-npr-4-thrs2.csv",
                  self.module_folder+"GenesExpressing-npr-11-thrs2.csv",
                  self.module_folder+"GenesExpressing-pdfr-1-thrs2.csv",
                  self.module_folder+"GenesExpressing-daf-2-thrs2.csv",
                  ]
          
        trans_rec_pairs = ["FLP-14,NPR-4",
                           "FLP-14,NPR-11",
                           "PDF-1,PDFR-1",
                           "INS-17,DAF-2"
                           ]
                           
        trans_exp_level = np.array([110634.0,110634.0,157972.0,1505.0])

        # Build expression levels                   
        exp_levels = np.zeros((len(fnames),self.n_neurons))
        for i_f in np.arange(len(fnames)):
            f = open(fnames[i_f],"r")
            lines = f.readlines()
            f.close()
            
            exp_levels_ = np.zeros(self.n_neurons)
            
            
            for line in lines[1:]:
                s = line.split(",")
                cell_id = s[1][1:-1]
                exp_level = float(s[2])
                
                cell_id = self.cengen_ids_conversion(cell_id)
                
                for cid in cell_id:
                    ai = self.ids_to_i(cid)
                    exp_levels[i_f,ai] += exp_level*trans_exp_level[i_f]
        
        if average:
            exp_levels = np.average(exp_levels,axis=0)
        
        return exp_levels    
    
    ###########
    # UTILITIES
    ###########
    
    def ds_to_file(self,folder,fname=None):
        '''Save the dataset list to a text file.
        
        Parameters
        ----------
        folder: str
            Destination folder.
        fname: str
            Destination filename. Default: Funatlas.ds_list_used_fname 
            (funatlas_list.txt)
        '''
        
        if fname is None:
            fname = self.ds_list_used_fname
        ds_tags = self.ds_tags if self.ds_tags is not None else ""
        ds_exclude_tags = self.ds_exclude_tags if self.ds_exclude_tags is not None else ""
        
        f = open(folder+fname,"w")
        for i_ds in range(len(self.ds_list)):
            ds = self.ds_list[i_ds]
            if self.ds_tags_lists is not None:
                f.write(ds+" # "+" ".join(self.ds_tags_lists[i_ds])+"\n")
        f.close()
    
    @classmethod
    def _is_bimodal1(cls,corr,kernels=None,return_stable_zeros=False):
        '''Returns an estimate on whether a distribution of correlations is 
        bimodal. It finds zeros of the derivative of the distribution that are
        stable with varying bandwidth of the derivative kernel, and determines
        whether the zeros fall in [-0.5,0.5] while the maximum outside of that
        range.
        
        Parameters
        ----------
        corr: array_like
            Distribution of the correlation. It is assumed to lie on the
            interval [-1,1].
        kernels: list of array_like (optional)
            Kernels to compute the derivative. If None, self.der_kernels is
            used. Default: None.
        return_stable_zeros: bool (optional)
            Whether to return also the stable zeros. Default: False.
            
        Returns
        -------
        is_bimodal: bool
            Whether the distribution is estimated to be bimodal.
        stable_zero1, stable_zero1b, stable_zero2: int
            Index of the stable zeros. See function definition for how they are
            calculated. Returned only if return_stable_zeros is True.
        
        '''
        
        if kernels is None:
            kernels =  cls.der_kernels
        
        n = len(corr)
        # Initialize list containing the zeros.
        zeros = []
        # Initialize array to store average.
        a2 = np.zeros_like(corr)
        # Prepare a warped version of the distribution.
        corrp = np.tile(corr,3)
        
        for der in kernels:
            # Calculate the derivative of the distribution with the given
            # kernel.
            a = np.convolve(der,corrp,mode="same")[n:-n]
            # Store it in the average.
            a2 += a
            
            absa = np.abs(a)
            # Smooth the derivative to get rid of most of the flat regions of 
            # contiguous zeros.
            box = np.ones(3)
            a = np.convolve(box,a,mode="same")
            
            # Find the zeros of this specific instance of the derivative and
            # append them to the common zero list.
            for i in np.argsort(absa):
                if i not in [0,n-1]:
                    if np.sign(a[i-1])!=np.sign(a[i+1]) and a[i+1]-a[i-1]!=0:
                        if i-1 not in zeros and i+1 not in zeros:
                            zeros.append(i)
        
        # Find the zeros of the average, which does not have all the quirks
        # of the derivatives calculated with the individual kernels.                    
        absa2 = np.abs(a2)
        maxabsa2=np.max(absa2)
        zeros_a2 = []
        for i in np.argsort(absa2):
            if i not in [0,n-1]:
                if np.sign(a2[i-1])!=np.sign(a2[i+1]):
                    if i-1 not in zeros_a2 and i+1 not in zeros_a2:
                        zeros_a2.append(i)
        
        # For the zeros of the average a2, pick the one that is in a minimum of
        # the distribution.
        zeros_a2_u, n_zeros_a2_u = np.unique(zeros_a2,return_counts=True)
        stable_zero2 = zeros_a2_u[np.argmin(corr[zeros_a2_u])]
        
        # For all the zeros, pick the one with the highest count.
        zeros_u, n_zeros_u = np.unique(zeros,return_counts=True)
        stable_zeros = zeros_u[np.argsort(n_zeros_u)]
        stable_zero1 = stable_zeros[-1]
        stable_zero1b = stable_zeros[-2]
        
        # The zeros must be within [-0.5,0.5] and the maximum outside of that
        # interval. 
        # Indices corresponding to -0.5 and +0.5 on a [-1,1] axis.
        i_m05 = n*0.25
        i_p05 = n*0.75
        is_bimodal =(i_m05<stable_zero1<i_p05 or i_m05<stable_zero1b<i_p05) and\
                    i_m05<stable_zero2<i_p05 and\
                    not i_m05<np.argmax(corr)<i_p05
        
        if return_stable_zeros:
            return is_bimodal,stable_zero1,stable_zero1b,stable_zero2
        else:
            return is_bimodal
            
    
    @staticmethod
    def threshold_to_sparseness(A,sparseness,absolute=True,max_i=100):
        amax = np.max(A)
        amin = np.min(A)
        
        a0 = amin
        a1 = amax
        
        if absolute:
            B = np.absolute(A)
        else:
            B = A
            
        totB = np.prod(B.shape)
        
        i = 0
        while True:
            th = 0.5*(a0+a1)
            sp = np.sum(B>th)/totB
            if np.absolute(sp-sparseness)/sparseness < 1e-2 or i>max_i:
                break
            
            if sp<sparseness:
                a1 = th
            else:
                a0 = th
            i+=1
            
        return th
            
    
    
    
    # Derivative kernels obtained with the following but stored to avoid a
    # dependency.   
    # ker_sizes = [5,7,9,11,13,15,19,23,27,31,39]
    # kernels = [sg.get_1D_filter(mu,3,1)[::-1] for mu in ker_sizes]      
    der_kernels = [[-0.08333333,0.6666667,0.,-0.6666667,0.08333333],
            [-0.08730159,0.265873,0.2301587,0.,-0.2301587,-0.265873,0.08730159],
            [-0.07239057,0.1195286,0.1624579,0.1060606,0.,-0.1060606,-0.1624579,
            -0.1195286,0.07239057],
            [-0.05827506,0.05710956,0.1033411,0.09770785,0.05749806,0.,
            -0.05749806,-0.09770785,-0.1033411,-0.05710956,0.05827506],
            [-0.04716117,0.02747253,0.06568432,0.07475857,0.06197969,0.03463203,
            0.,-0.03463203,-0.06197969,-0.07475857,-0.06568432,-0.02747253,
            0.04716117],
            [-0.03867102,0.01233271,0.042346,0.05486725,0.05339486,0.04142725,
            0.02246283,0.,-0.02246283,-0.04142725,-0.05339486,-0.05486725,
            -0.042346,-0.01233271,0.03867102],
            [-0.02711324,-0.00026582,0.01816931,0.02924368,0.03400882,
            0.03351628,0.02881759,0.02096429,0.01100791,0.,-0.01100791,
            -0.02096429,-0.02881759,-0.03351628,-0.03400882,-0.02924368,
            -0.01816931,0.00026582,0.02711324],
            [-0.01995541,-0.00412993,0.00769231,0.01591162,0.02092835,
            0.0231428,0.02295531,0.02076619,0.01697578,0.01198439
            ,0.00619236,0.,-0.00619236,-0.01198439,-0.01697578,-0.02076619,
            -0.02295531,-0.0231428,-0.02092835,-0.01591162,-0.00769231,
            0.00412993,0.01995541],
            [-0.01526485,-0.00516703,0.00278912,0.00878209,0.01299033,
            0.01559233,0.01676655,0.01669146,0.01554554,0.01350727,0.0107551,
            0.00746752,0.003823,0.,-0.003823,-0.00746752,-0.0107551,-0.01350727,
            -0.01554554,-0.01669146,-0.01676655,-0.01559233,-0.01299033,
            -0.00878209,-0.00278912,0.00516703,0.01526485],
            [-0.01203927,-0.00520743,0.00037698,0.00480307,0.00815994,
            0.01053669,0.01202242,0.01270624,0.01267724,0.01202453,0.0108372,
            0.00920437,0.00721514,0.00495859,0.00252385,0.,-0.00252385,
            -0.00495859,-0.00721514,-0.00920437,-0.0108372,-0.01202453,
            -0.01267724,-0.01270624,-0.01202242,-0.01053669,-0.00815994,
            -0.00480307,-0.00037698,0.00520743,0.01203927],
            [-0.00802585,-0.00447922,-0.00143923,0.00112228,0.00323346,
            0.00492245,0.00621739,0.00714643,0.00773772,0.0080194,0.00801962,
            0.00776653,0.00728827,0.00661299,0.00576883,0.00478394,0.00368646,
            0.00250455,0.00126635,0.,-0.00126635,-0.00250455,-0.00368646,
            -0.00478394,-0.00576883,-0.00661299,-0.00728827,-0.00776653,
            -0.00801962,-0.0080194,-0.00773772,-0.00714643,-0.00621739,
            -0.00492245,-0.00323346,-0.00112228,0.00143923,0.00447922,
            0.00802585]]
    

