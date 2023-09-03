import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker

ds_list = "/projects/LEIFER/francesco/funatlas_list.txt"
ds_tags = None
ds_exclude_tags = "mutant"
ds_exclude_i = []
inclall_occ = "--inclall-occ" in sys.argv
signal_range = None
smooth_mode = "sg_causal"
smooth_n = 13
smooth_poly = 1
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
nan_th = pp.Funatlas.nan_th
save = "--no-save" not in sys.argv

enforce_stim_crosscheck = "--enforce-stim-crosscheck" in sys.argv
merge_bilateral = "--no-merge-bilateral" not in sys.argv
req_auto_response = "--req-auto-response" in sys.argv
to_paper = "--to-paper" in sys.argv
plot = "--no-plot" not in sys.argv
figsize = (12,10)

for s in sys.argv:
    sa = s.split(":")
    if sa[0] == "--nan-th": nan_th = float(sa[1])
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    if sa[0] == "--ds-exclude-tags": 
        ds_exclude_tags=sa[1]
        if ds_exclude_tags == "None": ds_exclude_tags=None
    if sa[0] == "--ds-tags": ds_tags=sa[1]
    if sa[0] == "--ds-exclude-i": ds_exclude_i = [int(sb) for sb in sa[1].split(",")]
    if sa[0] == "--signal-range":
        sb = sa[1].split(",")
        signal_range = [int(sbi) for sbi in sb]
    if sa[0] == "--figsize": figsize = [int(sb) for sb in sa[1].split(",")]

# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, 
                 "smooth_mode": smooth_mode, 
                 "smooth_n": smooth_n, "smooth_poly": smooth_poly,                 
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only, 
                 "photobl_appl":True}

funa = pp.Funatlas.from_datasets(
                ds_list,
                merge_bilateral=merge_bilateral,
                merge_dorsoventral=False,
                merge_numbered=False,signal="green",
                signal_kwargs=signal_kwargs,
                ds_tags=ds_tags,ds_exclude_tags=ds_exclude_tags,
                enforce_stim_crosscheck=enforce_stim_crosscheck,
                verbose=False)

occ1, occ2 = funa.get_occurrence_matrix(inclall=inclall_occ,req_auto_response=req_auto_response)
# If occ2 needs to be filtered
occ1,occ2 = funa.filter_occ12_from_sysargv(occ2,sys.argv)
                 
occ3 = funa.get_observation_matrix_nanthresh(req_auto_response=req_auto_response)

if inclall_occ: inclall_occ2 = occ2
qvalues = funa.get_kolmogorov_smirnov_q(inclall_occ2)
qvalues_head = funa.reduce_to_head(qvalues)

# cumulative distribution of the qvalues
qvalues_flat = qvalues_head.flatten("C")
fig5 = plt.figure(figsize=(8, 4))
ax = plt.gca()
ax2 = ax.twinx()
n, bins, patches = ax.hist(qvalues_flat, 40, density=True, histtype='step',
                           cumulative=True, label='Qvalues', linewidth = 5)
poly = ax.findobj(plt.Polygon)[0]
vertices = poly.get_path().vertices

# Keep everything above y == 0. You can define this mask however
# you need, if you want to be more careful in your selection.
keep = vertices[:, 1] > 0

# Construct new polygon from these "good" vertices
new_poly = plt.Polygon(vertices[keep], closed=False, fill=False,
                       edgecolor=poly.get_edgecolor(),
                       linewidth=poly.get_linewidth())
poly.set_visible(False)
ax.add_artist(new_poly)
plt.draw()

ax2.hist(qvalues_flat, 40 , density=False, histtype='step',
                           cumulative=True, label='Qvalues')
poly = ax2.findobj(plt.Polygon)[0]
vertices = poly.get_path().vertices

# Keep everything above y == 0. You can define this mask however
# you need, if you want to be more careful in your selection.
keep = vertices[:, 1] > 0

# Construct new polygon from these "good" vertices
new_poly = plt.Polygon(vertices[keep], closed=False, fill=False,
                       edgecolor=poly.get_edgecolor(),
                       linewidth=poly.get_linewidth())
poly.set_visible(False)
ax2.add_artist(new_poly)
plt.draw()

ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
start, end = ax2.get_ylim()
ax.yaxis.set_major_locator(ticker.FixedLocator([0,0.5,1]))
ax2.yaxis.set_major_locator(ticker.FixedLocator([0,end/2,end]))
ax.set_xlabel('Qvalue',fontsize= 15)
ax.set_ylabel('Cumulative Density of Pairs',fontsize= 15)
ax2.set_ylabel('Number of Pairs',fontsize= 15)
ax.tick_params(axis='both', labelsize= 15)
ax2.tick_params(axis='y', labelsize= 15)
ax2.set_xticks([0,0.25,0.5])
#plt.xticks(np.arange(0, np.nanmax(qvalues_flat), step=np.nanmax(qvalues_flat)/2), fontsize= 15)
#plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/QvalueCDF.pdf", bbox_inches='tight')
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/QvalueCDF.pdf", bbox_inches='tight')

fig5.clf()


# cumulative distribution of the number of measurements or observations
occ3_head = funa.reduce_to_head(occ3)
occ3_flat = occ3_head.flatten("C")
fig6 = plt.figure(figsize=(5, 3))
ax = plt.gca()
ax2 = ax.twinx()
n, bins, patches = ax.hist(occ3_flat, occ3_head.max(), density=True, histtype='step',
                           cumulative=-1, label='Occ3', linewidth = 5)
poly = ax.findobj(plt.Polygon)[0]
vertices = poly.get_path().vertices
# Keep everything above y == 0. You can define this mask however
# you need, if you want to be more careful in your selection.
keep = vertices[:, 1] > 0
# Construct new polygon from these "good" vertices
new_poly = plt.Polygon(vertices[keep], closed=False, fill=False,
                       edgecolor=poly.get_edgecolor(),
                       linewidth=poly.get_linewidth())
poly.set_visible(False)
ax.add_artist(new_poly)
plt.draw()
ax2.hist(occ3_flat, occ3_head.max(), density=False, histtype='step',
                           cumulative=-1, label='Occ3')
poly = ax2.findobj(plt.Polygon)[0]
vertices = poly.get_path().vertices
# Keep everything above y == 0. You can define this mask however
# you need, if you want to be more careful in your selection.
keep = vertices[:, 1] > 0
# Construct new polygon from these "good" vertices
new_poly = plt.Polygon(vertices[keep], closed=False, fill=False,
                       edgecolor=poly.get_edgecolor(),
                       linewidth=poly.get_linewidth())
poly.set_visible(False)
ax2.add_artist(new_poly)
plt.draw()
start, end = ax2.get_ylim()
ax.yaxis.set_major_locator(ticker.FixedLocator([0,0.5,1]))
ax2.yaxis.set_major_locator(ticker.FixedLocator([0,end/2,end]))
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)
#plt.xticks(np.arange(0, 29, step=14), fontsize= 25)
ax.set_xticks([0,30,60])
ax.set_xlabel('Minimum Number of Observations of a Pair', fontsize= 15)
ax.set_ylabel('Cumulative Density of Pairs', fontsize= 15)
ax2.set_ylabel('Number of Pairs', fontsize= 15)
ax.tick_params(axis='both', labelsize= 25)
ax2.tick_params(axis='y', labelsize= 25)
#plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/Occ3CDF.pdf", bbox_inches='tight')
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/Occ3CDF.pdf", bbox_inches='tight')
fig6.clf()

fig6, ax = plt.subplots(figsize=(8, 4))
n, bins, patches = ax.hist(occ3_flat, 40, density=True, label='Occ3')
ax.set_xlabel('Number of Observations of a Pair')
ax.set_ylabel('Density')
#plt.savefig("/projects/LEIFER/Sophie/Figures/Robustness/Occ3PDF.pdf")
plt.savefig("/projects/LEIFER/francesco/funatlas/figures/paper/figS5/Occ3PDF.pdf")
fig6.clf()



