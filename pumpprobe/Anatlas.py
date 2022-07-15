import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pumpprobe as pp

class Anatlas:
    n = 302
    xyz_ids = []
    xyz = np.zeros((n,3))
    
    conn_chem = np.zeros((n,n))
    conn_gap = np.zeros((n,n))
    ids = []
    
    folder = os.path.dirname(pp.__file__)+"/"
    xyz_fname = "anatlas_neuron_positions.txt"
    conn_white_fname = "aconnectome.json"
    conn_white_ids_fname = "aconnectome_ids.txt"
    conn_witvliet_fname = "aconnectome_witvliet_2020_"
    conn_witvliet_fname_ext = ".csv"
        
    def __init__(self, source="witvliet-7"):
        
        self.load_aconnectome(source)
        
        f = open(self.folder+self.xyz_fname,'r')
        xyz_ids = f.readline()[1:-1].split(" ")
        f.close()
        
        xyz = np.loadtxt(self.folder+self.xyz_fname)[:,::-1].copy()
        
        self.missing_neurons = []
        for i in np.arange(len(xyz_ids)):
            try:
                j = self.ids.index(xyz_ids[i])
                self.xyz[j] = xyz[i]
            except:
                self.missing_neurons.append(xyz_ids[i])
                
        #print("Could not find the following neurons in the connectome: "+\
        #      str(self.missing_neurons))
        
    def load_aconnectome(self, source):
        if source == "white":
            self._load_aconnectome_white()
        elif source in ["witvliet-7","witvliet-8"]:
            self._load_aconnectome_witvliet(source)
    
    def _load_aconnectome_white(self):
        f = open(self.folder+self.conn_white_fname,"r")
        self.conn = json.load(f)
        f.close()
        
        # The connectome is indexed as [to,from]. AVA seems to confirm this
        # is the right indexing after transposing.
        self.conn_chem = np.copy((np.array(self.conn["chemical"])*\
                         np.array(self.conn["chemical_sign"])).T)
        self.conn_gap = np.copy(np.array(self.conn["electrical"]).T)
        
        f = open(self.folder+self.conn_white_ids_fname,'r')
        for l in f.readlines():
            neu_name = l.split("\t")[1]
            if neu_name[-1:] == "\n": neu_name = neu_name[:-1]
            self.ids.append(neu_name)
        f.close()
        
    def _load_aconnectome_witvliet(self, source):
        fname = self.conn_witvliet_fname+source[-1]+self.conn_witvliet_fname_ext
        f = open(self.folder+fname,'r')
        lines = f.readlines()
        f.close()
        
        # Build unique of list of neuron ids
        ids_mult = []
        for l in lines[1:]: #skip header
            ids_mult.append(l.split("\t")[0])
            ids_mult.append(l.split("\t")[1])
        self.ids = np.unique(ids_mult).tolist()
        
        self.n = len(self.ids)
        self.conn_chem = np.zeros((self.n,self.n))
        self.conn_gap = np.zeros((self.n,self.n))
        
        for l in lines[1:]:
            sl = l.split("\t")
            id_from = sl[0]
            id_to = sl[1]
            conn_type = sl[2]
            conn_n = int(sl[3])
            
            i_from = self.ids.index(id_from)
            i_to = self.ids.index(id_to)
            
            if conn_type == "chemical":
                self.conn_chem[i_to,i_from] = conn_n
            elif conn_type == "electrical":
                self.conn_gap[i_to,i_from] = conn_n
                
    def get_paths(self,i,j,max_n_hops=1,return_ids=False,exclude_self_loops=True):
        # You could add a check with a scalar Dyson equation to see if there
        # is a path at all. You can use the effective anatomical connectivity
        # at all steps to help speed up the process.
        paths_final = []
        
        paths_all = []
        # Populate paths_all with all the 1-hop connections that send signals
        # into i.
        for q in np.arange(self.conn_chem.shape[0]):
            if self.conn_chem[i,q] != 0 or self.conn_gap[i,q] !=0:
                paths_all.append([i,q])

        for h in np.arange(max_n_hops):
            paths_all_new = []
            for p in paths_all:
                if p[-1] == j: 
                    # Arrived at j
                    paths_final.append(p)
                elif h!=max_n_hops-1:
                    # Iterate over all the connections and add a hop
                    for q in np.arange(self.conn_chem.shape[0]):
                        if (self.conn_chem[p[-1],q] != 0 or self.conn_gap[p[-1],q] !=0) and not (exclude_self_loops and q==i):
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
                    paths_final_ids[i_p][q] = self.ids[paths_final_ids[i_p][q]]
            
            return paths_final, paths_final_ids
        else:
            return paths_final
        
    
    def plot_worm(self):
        fig = self._get_figure()
        ax1 = fig.add_subplot(111)
        
        ax1.plot(self.xyz[:,0],self.xyz[:,1],'ob')
        plt.show()
        
    def plot_connected(self,neu_i=0,neu_id=None,head_only=False,labels=False):
        if neu_id is not None:
            try: 
                neu_i = self.ids.index(neu_id)
            except:
                try:
                    print(neu_id+" not found. ",end="")
                    neu_id += "L"
                    print("Trying with "+neu_id+".")
                    neu_i = self.ids.index(neu_id)
                except:
                    print("Neuron not found.")
                    quit()
                
        else:
            neu_id = self.ids[neu_i]
        
        fig = self._get_figure()
        ax1 = fig.add_subplot(111)
        
        ax1.plot(self.xyz[:,0],self.xyz[:,1],'o',c='k',alpha=0.5,markersize=1)
        
        conn_i = neu_i
        chem = np.where(self.conn_chem[:,conn_i]!=0)[0]
        if len(chem)>0: max_n_chem = np.max(self.conn_chem[chem,conn_i])
        else: max_n_chem=1
        
        for j in chem:
            if self.conn_chem[j,conn_i]>0: c='b'
            else: c='r'
            alpha = abs(self.conn_chem[j,conn_i]/max_n_chem)
            #xyz_i_tmp = self.xyz_ids.index(self.ids[j])
            x = self.xyz[j,0]
            y = self.xyz[j,1]
            ax1.plot(x,y,'o',c=c,alpha=alpha)
            if labels:
                ax1.text(x,y,self.ids[j])
            
        gap = np.where(self.conn_gap[:,conn_i]!=0)[0]
        if len(gap)>0: max_n_gap = np.max(self.conn_gap[gap,conn_i])
        else: max_n_gap=1
        
        for j in gap:
            c = 'g'
            alpha = self.conn_gap[j,conn_i]/max_n_gap
            #xyz_i_tmp = self.xyz_ids.index(self.ids[j])
            x = self.xyz[j,0]
            y = self.xyz[j,1]
            ax1.plot(x,y,'o',c=c,alpha=alpha)
            if labels:
                ax1.text(x,y,self.ids[j])
         
        ax1.plot(self.xyz[neu_i,0],self.xyz[neu_i,1],'*',c='orange')
         
        ax1.set_title("From "+neu_id)
        if head_only:
            ax1.set_xlim(0,0.65)
            ax1.set_ylim(-3.3,-1.7)
            
        plt.show()
        
    def plot_connected_avg(self, head_only=True):
    
        fig = self._get_figure()
        ax1 = fig.add_subplot(111)
        
        max_n_chem = np.median(np.sort(np.ravel(self.conn_chem))[-10:])
        max_n_gap = np.median(np.sort(np.ravel(self.conn_gap))[-10:])
        
        for k in np.arange(self.conn_chem.shape[0]):
            xyz_i_from = k#self.xyz_ids.index(self.ids[k])
            x_from = self.xyz[xyz_i_from,0]
            y_from = self.xyz[xyz_i_from,1]
            
            if head_only:
                if not( (0 < x_from < 0.65) and (-3.3 < y_from < -1.7) ):
                    continue
            
            chem = np.where(self.conn_chem[:,k]!=0)[0]
            for j in chem:
                if self.conn_chem[j,k]>0: c='b'
                else: c='r'
                alpha = np.clip(abs(self.conn_chem[j,k]/max_n_chem),0,1)
                xyz_i_to = j#self.xyz_ids.index(self.ids[j])
                x = self.xyz[xyz_i_to,0] - self.xyz[xyz_i_from,0]
                y = self.xyz[xyz_i_to,1] - self.xyz[xyz_i_from,1]
                ax1.plot(x,y,'o',c=c,alpha=alpha)
            
            gap = np.where(self.conn_gap[:,k]!=0)[0]
            for j in gap:
                c = 'g'
                alpha = np.clip(self.conn_gap[j,k]/max_n_gap,0,1)
                xyz_i_to = j#self.xyz_ids.index(self.ids[j])
                x = self.xyz[xyz_i_to,0] - self.xyz[xyz_i_from,0]
                y = self.xyz[xyz_i_to,1] - self.xyz[xyz_i_from,1]
                ax1.plot(x,y,'o',c=c,alpha=alpha)
        
        if head_only:
            ax1.set_xlim(-0.65,0.65)
            ax1.set_ylim(-1.6,1.6)
        
        plt.show()
        
    @staticmethod
    def _get_figure():
        cfn = plt.gcf().number
        if len(plt.gcf().axes)!=0: cfn += 1    
        fig = plt.figure(cfn)
        
        return fig
        
    
