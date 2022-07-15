import numpy as np, matplotlib.pyplot as plt
import pumpprobe as pp

fname = "external_data/GenesExpressing-unc-7-unc-9-inx-_-eat-5-thrs2.csv"
funa = pp.Funatlas(merge_bilateral=True)
funa.load_aconnectome_from_file(gap_th=0)

f = open(fname,"r")
lines = f.readlines()
f.close()
ids = lines[0].split(",")[4:]
exp_levels = np.zeros((len(ids),len(lines)-1))
genes = []


for il in np.arange(len(lines)-1):
    l = lines[il+1]
    a=l.split(",")
    genes.append(a[1])
    exp_levels[:,il] = [float(b) for b in a[4:]]

neurons = ["AVE","AVA","M1","RIG","RIH","RME_DV","RMD_DV","URY","CEP","OLQ","I2","I4","IL2_DV"]
neurons2= ["RME_", "RMDV_", "URYD_", "CEPD_", "OLQD_", "I2_", "I4", "IL2V_"]

unc7_i = genes.index("unc-7")
unc9_i = genes.index("unc-9")

unc7_9_to_others = np.zeros_like(neurons,dtype=float)
other_exp_levels=np.delete(exp_levels,(unc7_i,unc9_i),axis=-1)

AVE_ai = funa.ids_to_i("AVE")
I1_ai = funa.ids_to_i("I1")

#funa.load_innexin_expression_from_file()
inx_exp_levels, inx_genes, funaunc7_9_to_others = funa.get_inx_exp_levels()
fr = funa.get_fractional_gap_inx_mutants()
print("fraction of gap junctions left in unc-9 unc-7 mutant")
print(funaunc7_9_to_others[funa.ids_to_i("I1")])
#print("URYD RIG",fr[funa.ids_to_i("URYD"),funa.ids_to_i("RIG")])
#print("OLQV RIG",fr[funa.ids_to_i("OLQV"),funa.ids_to_i("RIG")])
#print("CEPD RIH",fr[funa.ids_to_i("CEPD"),funa.ids_to_i("RIH")])
#print("RMDV AVE",fr[funa.ids_to_i("RMDV"),funa.ids_to_i("AVE")])
#print("RME AVE",fr[funa.ids_to_i("RME"),funa.ids_to_i("AVE")])
print("IL1V_ RME",fr[funa.ids_to_i("IL1V_"),funa.ids_to_i("RME")])
print("I3 M2",fr[funa.ids_to_i("I3"),funa.ids_to_i("M2")])
print("M3 M2",fr[funa.ids_to_i("M3"),funa.ids_to_i("M2")])
print("I2 M1",fr[funa.ids_to_i("I2"),funa.ids_to_i("M1")])
print("I4 M1",fr[funa.ids_to_i("I4"),funa.ids_to_i("M1")])

for i in np.arange(len(neurons)):
    j = ids.index(neurons[i])
    ai = funa.ids_to_i(neurons[i])
    
    unc7_9_to_others[i] = (exp_levels[j,unc7_i]+exp_levels[j,unc9_i])/np.sum(exp_levels[j])
    
    print(neurons[i],"\t",np.around(funaunc7_9_to_others[ai],2))   
    print(neurons[i],"\t",np.around(unc7_9_to_others[i],2),"\t",int(np.sum(other_exp_levels[j])))
