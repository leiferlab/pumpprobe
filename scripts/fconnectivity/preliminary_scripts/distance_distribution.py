import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
req_auto_response = True#"--req-auto-response" in sys.argv

funa = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags=None,ds_exclude_tags="mutant",
                verbose=False)
                
funa_unc31 = pp.Funatlas.from_datasets(
                ds_list,merge_bilateral=True,merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                ds_tags="unc31",ds_exclude_tags=None,
                verbose=False)
   
aconn = funa.get_boolean_aconn()
esconn = funa.get_esconn()

occ1, occ2 = funa.get_occurrence_matrix(req_auto_response=req_auto_response)
dist1,dist2 = funa.get_distances(occ2)
dist3,dist4,resp3,resp4 = funa.get_distances_resp_nonresp(req_auto_response)

dist3_aconn,_,resp3_aconn,_ = funa.get_distances_from_conn(aconn)
dist3_esconn,_,resp3_esconn,_ = funa.get_distances_from_conn(esconn)

dist3_unc31,dist4_unc31,resp3_unc31,resp4_unc31 = funa_unc31.get_distances_resp_nonresp(req_auto_response)

fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
ax.hist(np.ravel(dist1)*0.42,bins=100,range=(0,150))
ax.set_xlabel("distance (um)")

fig2 = plt.figure(2)
ax = fig2.add_subplot(111)
ax.hist(dist3[resp3]*0.42,bins=100,range=(0,150),label="responding",alpha=0.5,color="#1f77b4")
ax.hist(dist3[~resp3]*0.42,bins=100,range=(0,150),label="not responding",alpha=0.5,color="#ff7f0e")
ax.set_xlabel("distance (um)")
ax.set_ylabel("number of neurons")
ax.set_title("In our data")
ax.legend()
pp.provstamp(ax,-.05,-.05," ".join(sys.argv))
fig2.tight_layout()

fig3 = plt.figure(3)
ax = fig3.add_subplot(111)
ax.hist(dist3[resp3]*0.42,bins=100,range=(0,150),label="responding",alpha=0.5,color="#1f77b4")
ax.hist(dist3_aconn[resp3_aconn]*0.42,bins=100,range=(0,150),label="responding aconn",histtype='step',alpha=0.5,color="#2ca02c")
ax.hist(dist3_esconn[resp3_esconn]*0.42,bins=100,range=(0,150),label="responding esconn",histtype='step',alpha=0.5,color="#9467bd")
ax.set_xlabel("distance (um)")
ax.set_ylabel("number of neurons")
ax.set_title("Responding in data vs in anatomical and extrasynaptic connectome")
ax.legend()
pp.provstamp(ax,-.05,-.05," ".join(sys.argv))
fig3.tight_layout()

fig4 = plt.figure(4)
ax = fig4.add_subplot(111)
ax.hist(dist3[resp3]*0.42,bins=100,range=(0,150),label="responding",alpha=0.5,density=True,color="#1f77b4")
ax.hist(dist3[~resp3]*0.42,bins=100,range=(0,150),label="not responding",alpha=0.5,density=True,color="#ff7f0e")
ax.hist(dist3_aconn[resp3_aconn]*0.42,bins=100,range=(0,150),label="responding aconn",histtype='step',alpha=0.5,density=True,color="#2ca02c")
ax.hist(dist3_aconn[~resp3_aconn]*0.42,bins=100,range=(0,150),label="not responding aconn",histtype='step',alpha=0.5,density=True,color="#d62728")
ax.hist(dist3_esconn[resp3_esconn]*0.42,bins=100,range=(0,150),label="responding esconn",histtype='step',alpha=0.5,density=True,color="#9467bd")
ax.hist(dist3_esconn[~resp3_esconn]*0.42,bins=100,range=(0,150),label="not esponding esconn",histtype='step',alpha=0.5,density=True,color="#8c564b")
ax.set_xlabel("distance (um)")
ax.set_ylabel("density")
ax.set_title("Responding/nonresponding in data vs in anatomical and extrasynaptic connectome")
ax.legend()
pp.provstamp(ax,-.05,-.05," ".join(sys.argv))
fig4.tight_layout()

fig5 = plt.figure(5)
ax = fig5.add_subplot(111)
hist_dist3,bin_edges = np.histogram(dist3[resp3]*0.42,bins=100)
hist_dist3_unc31,bin_edges_unc31 = np.histogram(dist3_unc31[resp3_unc31]*0.42,bins=100)
cdf = np.cumsum(hist_dist3)
cdf = cdf/np.max(cdf)
cdf_unc31 = np.cumsum(hist_dist3_unc31)
cdf_unc31 = cdf_unc31/np.max(cdf_unc31)
ax.plot(bin_edges[:-1],cdf,label="wt")
ax.plot(bin_edges_unc31[:-1],cdf_unc31,label="unc-31")
ax.set_xlabel("distance (um)")
ax.set_ylabel("cdf")
ax.legend()
pp.provstamp(ax,-.05,-.05," ".join(sys.argv))
fig5.tight_layout()

plt.show()
