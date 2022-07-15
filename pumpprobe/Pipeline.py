import os
from datetime import datetime
import json
import numpy as np

class Pipeline:
    logbook_fname = "analysis.log"
    logbook_f = None
    pre_log = ""
    full_command = ""
    end = False

    ################
    # Default values
    ################
    config = {}
    config['folder'] = None
    config['legacy'] = False
    config['segment'] = True
    config['match'] = True
    config['match_method'] = 'dsmm'
    config['add_neurons'] = False
    config['merge_reference_neurons'] = False
    config['matchless_tracking'] = False
    config['alignment_check'] = True
    config['flash_threshold'] = 20
    config['signal_membrane'] = False
    config['flagged_frames'] = []
    config['flagged_volumes'] = []
    config['signal_method'] = "weightedMask"
    config['box_size'] = [5,5,5]
    config["signal_from_stabilized"] = True
    config["stabilize"] = True
    config["pixel_offset"] = 100
    config["shift_g"] = [0,0]
    config['selected_frame_box'] = None

    segm_params = {}
    segm_params['threshold'] = 0.1
    segm_params['A_threshold'] = 110
    segm_params['A_threshold_mode'] = "manual"
    segm_params['A_threshold_multiplier'] = 1
    segm_params['channel'] = 0
    segm_params['blur'] = 0.7
    segm_params['dil_size'] = 5
    segm_params['check_planes_n'] = 7
    config['segm_params'] = segm_params
    
    reg_params = {}
    reg_params['reg_type'] = 'dsmm'
    reg_params['insist'] = True
    reg_params['beta'] = 3.0
    reg_params['lambda'] = 5.0
    reg_params['neighbor_cutoff'] = 10.0
    reg_params['max_iter'] = 100
    reg_params['ref'] = "median"
    config['reg_params'] = reg_params
    
    def __init__(self, calling_script, argv = None, folder = None):
        '''The constructor parses the sys.argv-style input. The constructor does
        not "start" the pipeline, i.e. write the configuration to the logbook, 
        so that the parameters can be set in the script after sys.argv has been 
        parsed.
        
        Parameters
        ----------
        calling_script: string
            Name describing the calling script.
        argv: list of strings
            sys.argv-style inputs. sys.argv can be passed. Default: None.
        folder: string
            Folder containing the data.
        '''
        self.calling_script = calling_script
        
        if argv is not None:
            self.parse_input(argv)
        
        if folder is not None:
            if folder[-1]!="/": folder+="/"
            self.config["folder"] = folder
        
        if not os.path.isdir(self.config["folder"]):
            print("Pipeline says: invalid folder.")
        
    def __del__(self):
        try:
            self.logbook_f.close()
        except:
            pass
            
    def log(self, s = None, print_to_terminal = True):
        '''Write an entry in the logbook and, if requested, print the same
        entry to terminal. In the logbook, the entry will start with the 
        current time.
        
        Parameters
        ----------
        s: string
            Text of the entry.
        print_to_terminal: boolean
            If True, the entry will also be printed to terminal. Default: True.

        Returns
        -------
        None
        '''
        if s is not None:
            now = datetime.now()
            dt = now.strftime("%Y-%m-%d %H:%M:%S "+self.calling_script+": ")
            if self.logbook_f is not None:
                self.logbook_f.write(dt+s+"\n")
            else:
                self.pre_log += (dt+s+"\n")
            
            if print_to_terminal:
                print(s)
            
    def start(self, config_to_log = True):
        '''Write entry to logbook with the chosen configuration. This is not
        executed by the constructor so that the parameters can be set in the 
        script after sys.argv has been parsed.
        
        Parameters
        ----------
        config_to_log: boolean
            If True, the configuration is written to the logbook.
            
        Returns
        -------
        None.       
        '''
        #self.logbook_f = open(self.config['folder']+self.logbook_fname, "a")
        self.open_logbook_f()
        self.log(self.pre_log, False)
        
        self.log("",False)
        self.log("## STARTING THE PIPELINE", False)
        self.log("Command used: "+self.full_command, False)
            
        if config_to_log:
            string = json.dumps(self.config)
            self.log("Using the following configuration: "+string, False)
            
    def open_logbook_f(self):
        self.logbook_f = open(self.config['folder']+self.logbook_fname, "a")
    
    def end(self):
        self.end = True
            
    def parse_input(self, argv):
        '''Parses the sys.argv-style input and updates the configuration
        accordingly. If --help is in the input, prints the help string.
        
        Parameters
        ----------
        argv: list of strings
            sys.argv-style input.        
        '''
        if "--help" in argv:
            self.print_help()
            quit()
        
        self.full_command = "python "+" ".join(argv)
    
        folder = argv[1]
        if folder[-1]!="/": folder+="/"
        self.config['folder'] = folder
    
        for a in argv:
            s = a.split(":")
        
            if len(s) == 1:
                if s[0] == "--legacy":
                    self.config["legacy"] = True
                elif s[0] == "--no-alignment-check":
                    self.config['alignment_check'] = False
                elif s[0] == "--no-segment":
                    self.config['segment'] = False
                    self.log("Skipping segmentation. Using existing results.")
                elif s[0] == "--add-neurons": 
                    self.config['add_neurons'] = True
                    self.config['matchless_tracking'] = True
                elif s[0] == "--merge-reference-neurons":
                    self.config['merge_reference_neurons'] = True
                elif s[0] == "--matchless-tracking":
                    self.config['matchless_tracking'] = True
                elif s[0] == "--no-match":
                    self.config['match'] = False
                    self.log("Skipping matching. Using existing results.")
                elif s[0] == "--no-match-insist":
                    self.config['reg_params']['insist'] = False
                elif s[0] == "--membrane-signal" or s[0] == "--signal-membrane":
                    self.config['signal_membrane'] = True
                    self.log("The signal will be extracted from the membranes.")
                elif s[0] == "--signal-unstabilized":
                    self.config["signal_from_stabilized"] = False
                    self.log("The signal will be extracted from the unstabilized neural positions.")
                elif s[0] == "--no-stabilize":
                    self.config["stabilize"] = False
                elif s[0] == self.calling_script or s[0] == self.config["folder"]:
                    pass
                else:
                    self.log(s[0]+" does not match any allowed parameter. --help for parameter list.")
                    #quit()
                    
            if len(s) == 2:
                ############################
                # Flagged volumes and frames
                ############################
                if s[0] == "--flagged-volumes":
                    self.config["flagged_volumes"] = []
                    st = s[1].split(",")
                    for su in st:
                        sv = su.split("-")
                        if len(sv)==1:
                            self.config["flagged_volumes"].append(int(su))
                        if len(sv)>1:
                            for fl in range(int(sv[0]),int(sv[1])):
                                self.config["flagged_volumes"].append(fl)
                
                elif s[0] == "--flagged-frames":
                    self.config["flagged_frames"] = []
                    st = s[1].split(",")
                    for su in st:
                        sv = su.split("-")
                        if len(sv)==1:
                            self.config["flagged_frames"].append(int(su))
                        if len(sv)>1:
                            for fl in range(int(sv[0]),int(sv[1])):
                                self.config["flagged_frames"].append(fl)

                elif s[0] == "--signal-method":
                    if s[1] in ["box","weightedMask"]:
                        self.config['signal_method'] = s[1]
                    else:
                        self.log("Invalid signal extraction method.")
                        quit()
                elif s[0] == "--box-size":
                    self.config["box_size"] = [int(a) for a in s[1].split("-")]
                    
                elif s[0] == "--pixel-offset":
                    self.config["pixel_offset"] = int(s[1])
                    
                elif s[0] == "--shift-g":
                    self.config["shift_g"] = [int(a) for a in s[1].split("-")]
                    
                elif s[0] == "--selected-frame-box":
                    if s[1] == "auto":
                        self.config["selected_frame_box"] = "auto"
                    else:
                        self.config["selected_frame_box"] = [int(a) for a in s[1].split("-")]
                
                #######
                # Flash
                #######
                elif s[0] == "--flash-threshold":
                    self.config["flash_threshold"] = float(s[1])
                
                ##############
                # Segmentation
                ##############
                elif s[0] == "--threshold" or s[0] == "--segm-threshold":
                    self.config['segm_params']['threshold'] = float(s[1])
                elif s[0] == "--A-threshold" or s[0] == "--segm-A-threshold":
                    sa = s[1].split("*")
                    if sa[0] == "auto":
                        self.config['segm_params']['A_threshold'] = None
                        self.config['segm_params']['A_threshold_mode'] = "auto"
                        if len(sa)==1:
                            self.config['segm_params']['A_threshold_multiplier'] = 1
                        else:
                            self.config['segm_params']['A_threshold_multiplier'] = float(sa[1])
                    else:    
                        self.config['segm_params']['A_threshold'] = int(s[1])
                elif s[0] == "--segm-channel":
                    self.config['segm_params']['channel'] = int(s[1])
                elif s[0] == "--segm-blur":
                    self.config['segm_params']['blur'] = float(s[1])
                elif s[0] == "--segm-dil-size":
                    self.config['segm_params']['dil_size'] = int(s[1])
                elif s[0] == "--segm-check-planes-n":
                    self.config['segm_params']['check_planes_n'] = float(s[1])
                    
                ##########
                # Matching
                ##########
                elif s[0] == "--match-method":
                    self.config['match_method'] = s[1]
                
                ##############
                # Registration
                ##############
                elif s[0] == "--reg": 
                    self.config['reg_params']['reg_type'] = s[1]
                elif s[0] == "--ref":
                    if s[1] in ["median","most"] or s[1].split(".")[0] in ["most"]:
                        self.config['reg_params']['ref'] = s[1]
                    else:
                        try:
                            self.config['reg_params']['ref'] = int(s[1])
                        except:
                            print("Invalid value passed as ref. Using",self.config['reg_params']['ref'])
                elif s[0] == "--dsmm-beta":
                    self.config['reg_params']['beta'] = float(s[1])
                elif s[0] == "--dsmm-lambda":
                    self.config['reg_params']['lambda'] = float(s[1])
                elif s[0] == "--dsmm-neighbor-cutoff":
                    self.config['reg_params']['neighbor_cutoff'] = float(s[1])
                elif s[0] == "--dsmm-max-iter":
                    self.config['reg_params']['max_iter'] = int(s[1])
                else:
                    self.log(s[0]+" does not match any allowed parameter. --help for parameter list.")
                    #quit()
                    
    
    @staticmethod
    def get_free_memory(from_meminfo=True):
        '''Determine how much RAM is available. To be used mostly when the 
        pipeline is not run through a job scheduling system.
        
        Parameters
        ----------
        from_meminfo: boolean (optional)
            If True, the free memory is determined from the file /proc/meminfo.
            Default: True
            
        Returns
        -------
        free_memory: float
            Free memory in GB. 
        '''
        if from_meminfo:
            f = open("/proc/meminfo","r")
            Line = f.readlines()

            for line in Line:
                sline = line.split(" ")
                if sline[0]=="MemFree:":
                    free_memory = float(sline[-2])/1024./1024.
                    break
            f.close()
        else:
            free_memory: 0.0
        return free_memory
    
    @staticmethod    
    def get_max_job_memory(fraction=0.05,**kwargs):
        '''Returns the maximum memory the script should use, given the
        available RAM on the system.
        
        Parameters
        ----------
        fraction: float (optional)
            Fraction of the available RAM to use. Default: 0.05
        **kwargs: parameters for get_free_memory() (optional)
        
        Returns
        -------
        max_job_memory: float
            Maximum memory the script should use, in Bytes.
        '''
        free_memory = Pipeline.get_free_memory(**kwargs)
        max_job_memory = np.around(free_memory*fraction*(2**30),2)
        
        return max_job_memory
        
    @staticmethod        
    def print_help():
        '''Print help string.'''
        s = "General\n"+\
            "\t--legacy\n"+\
            "\t--no-alignment-check\n\t--no-segment\n\t--add-neurons\n"+\
            "\t--matchless-tracking\n\t--no-match\n\t--no-match-insist\n"+\
            "\t--membrane-signal or --signal-membrane\n"+\
            "\t--flash-threshold:float (Default:20)\n"+\
            "\t--flagged-frames:comma separated floats (or float-float for ranges) (Default: None)\n"+\
            "\t--flagged-volumes:comma separated floats (or float-float for ranges) (Default: None)\n"+\
            "\t--signal-method:str (Possible values: box,weightedMask. Default: weightedMask)\n"+\
            "\t--signal-unstabilized\n"+\
            "\t--stabilize\n"+\
            "\t--box-size:int-int-int (See wormdatamodel.signal.extraction. Default 5-5-5)\n"+\
            "\t--pixel-offset: int (Default: 100)\n"+\
            "\t--shift-g\n"+\
            "\t--selected-frame-box\n"+\
            "Segmentation\n"+\
            "\t--threshold:float or --segm-threshold:float (Default:0.1)\n"+\
            "\t--A-threshold:int or --segm-A-threshold:int threshold on the raw image values, not curvature. Can be auto*multiplier. (Default: 110)\n"+\
            "\t--segm-channel:int (Possible values: 0,1. Default: 0)\n"+\
            "\t--segm-blur:float (Default: 0.7)\n"+\
            "\t--segm-dil-size:int (odd, Default:5)\n"+\
            "\t--segm-check-planes-n:int (Possible values: 5,7. Default: 7)\n"+\
            "Registration and matching\n"+\
            "\t--ref:string or int (Possible values: median, most, or explicit index of reference volume. Default: median)\n"+\
            "\t--match-method: string (Possible values: dsmm, fdlc. Default: dsmm)\n"+\
            "\t--reg-type: string (Possible values: centroid, displacement, dsmm, none. Default: dsmm)\n"+\
            "\t--dsmm-beta:float (Default:3.0)\n"+\
            "\t--dsmm-lambda:float (Default: 5.0)\n"+\
            "\t--dsmm-neighbor-cutoff:float (Default: 10.0)\n"+\
            "\t--dsmm-max-iter:int (Default 100)\n"
        print(s)
