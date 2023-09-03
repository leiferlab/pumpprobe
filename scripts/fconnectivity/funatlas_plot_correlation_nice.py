import numpy as np, matplotlib.pyplot as plt, sys
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

to_paper = "--to-paper" in sys.argv
compreal = "comp"
fname_add = ""
for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--real": compreal = "real"
    if sa[0] == "--merged-L-R": fname_add = "_merged_"

print("Using the comp/real responses",compreal)

pair_comp_resp_coeff = np.loadtxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations"+fname_add+"_pair_"+compreal+"_resp_coeff.txt")
all_comp_resp_corr = np.loadtxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations"+fname_add+"_all_"+compreal+"_resp_corr.txt")

nbins = 30
fig = plt.figure(1,figsize=(3,2))
ax = fig.add_subplot(111)
x0,y0,_ = ax.hist(pair_comp_resp_coeff, bins = 30, alpha = 0.5, density = "True",label="Responses generated from\nthe same pair of neurons")
x1,y1,_ = ax.hist(all_comp_resp_corr, bins = 30, alpha = 0.5, density = "True",label="All generated responses\nshuffled")
ax.set_xlabel("Correlation coefficient",fontsize=14)
ax.set_ylabel("Prob Density",fontsize=14)
ax.legend(fontsize=7,loc=2)
ax.set_xticks(np.arange(-1, 1.5, step=0.5))
ax.set_yticks([0,1.5,3])

fig.tight_layout()
fig.savefig("/projects/LEIFER/francesco/funatlas/diagnostics/allcorrelations_nice.png",dpi=300)
if to_paper:
    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations_nice_A_n.txt",x0)
    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations_nice_A_bins.txt",y0)
    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations_nice_B_n.txt",x1)
    np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations_nice_B_bins.txt",y1)
    
    n1 = len(x0)
    n2 = len(x1)
    print(n1,len(x0),n2,len(x1))
    assert n1==n2
    f = open("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations_nice.txt", "w")
    f.write("bin_edges_wt,y_wt,bin_edges_unc31,y_unc31\n")
    s = ""
    for i in np.arange(n1):
        s += str(x0[i])+","+str(y0[i])+","+str(x1[i])+","+str(y1[i])+"\n"
    s = s[:-1]
    f.write(s)
    f.close()
    
    
    fig.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations_nice.pdf",bbox_inches="tight")
plt.show()
