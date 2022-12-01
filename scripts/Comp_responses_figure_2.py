import numpy as np, matplotlib.pyplot as plt, sys
import pumpprobe as pp, wormdatamodel as wormdm, wormbrain as wormb
from scipy import stats

# List of datasets
fname = "/projects/LEIFER/Sophie/funatla_list_sophie.txt"

file = open(fname, "r")
nr_of_datasets = sum(1 for line in file)
file.close()

# Read IDs of stimulated neuron (j) and responding neuron (i)
i = j = ""
all_pairs = "--all-pairs" in sys.argv
use_kernels = "--use-kernels" in sys.argv
merge_LR = "--merge-L-R" in sys.argv
plots_make = "--make-plots" in sys.argv
signal = "green"
matchless_nan_th = None
matchless_nan_th_from_file = "--matchless-nan-th-from-file" in sys.argv
matchless_nan_th_added_only = "--matchless-nan-th-added-only" in sys.argv
normalize = ""
for s in sys.argv:
    sa = s.split(":")
    if sa[0] in ["-j","--j"] : jid = sa[1]
    if sa[0] in ["-i","--i"]: iid = sa[1]
    if sa[0] == "--signal": signal = sa[1]
    if sa[0] == "--matchless-nan-th": matchless_nan_th = float(sa[1])
    if sa[0] == "--normalize": normalize = sa[1]
if not all_pairs and (i=="" or j==""): print("Select i and j.");quit()


# Prepare kwargs for signal preprocessing (to be passed to Funatlas, so that
# it can internally apply the preprocessing to the Signal objects).
signal_kwargs = {"remove_spikes": True,  "smooth": True, "smooth_mode": "sg",
                 "smooth_n": 13, "smooth_poly": 1,
                 "matchless_nan_th_from_file": matchless_nan_th_from_file,
                 "matchless_nan_th": matchless_nan_th,
                 "matchless_nan_th_added_only": matchless_nan_th_added_only,
                 "photobl_appl":True,}


# Build Funatlas object
funatlas = pp.Funatlas.from_datasets(
                fname,merge_bilateral=merge_LR,merge_dorsoventral=False,
                merge_numbered=False,signal=signal,signal_kwargs=signal_kwargs)


# Occurence matrix. occ1[i,j] is the number of occurrences of the response of i
# following a stimulation of j. occ2[i,j] is a dictionary containing details
# to extract the activities from the signal objects.
occ1, occ2 = funatlas.get_occurrence_matrix(req_auto_response=True)

#print(occ1)

iids = jids = funatlas.neuron_ids  #gives ids for each global index


all_kernels_ec = []
all_fitted_stims_ec = []
kernels_by_pair_ec = {}
fitted_stims_by_pair_ec = {}
all_dt = []
dt_by_pair = {}
pair_by_pair = {}

pair_ind = 1
for iid in iids:
	for jid in jids:

		# Convert the requested IDs to atlas-indices.
		i,j = funatlas.ids_to_i([iid,jid])
		if i<0: print(iid,"not found. Check approximations.")
		if j<0: print(jid,"not found. Check approximations.")


		if occ1[i,j]>1: #only for pairs with more than 2 responses
			#fig = plt.figure(1,figsize=(15,8))
			#ax1 = fig.add_subplot(121)
			#ax2 = fig.add_subplot(122)

			kernels_by_pair_ec[str(pair_ind)] = []
			fitted_stims_by_pair_ec[str(pair_ind)] = []
			dt_by_pair[str(pair_ind)] = []
			pair_by_pair[str(pair_ind)] = [iid, jid]


			occ_ind = 1
			for occ in occ2[i,j]: #for each of the response and stimulus instances
				ds = occ["ds"] #dataset number
				ie = occ["stim"] #stimulation number
				resp_i = occ["resp_neu_i"] #number of the responding neuron
				#print("number of stims for this pair", occ1[i,j])
				#print("iid",iid)
				#print("jid",jid)



				# Build the time axis
				i0 = funatlas.fconn[ds].i0s[ie]
				i1 = funatlas.fconn[ds].i1s[ie]
				Dt = funatlas.fconn[ds].Dt
				next_stim_in_how_many = funatlas.fconn[ds].next_stim_after_n_vol
				shift_vol = funatlas.fconn[ds].shift_vol
				time_trace = (np.arange(i1-i0)-shift_vol)*Dt
				#print(next_stim_in_how_many)
				time = (np.arange(next_stim_in_how_many[ie]))*Dt #this is the time axis starting from 0, from the time_0_vol
				#print(time)
				stim_j = funatlas.fconn[ds].stim_neurons[ie]


				# try:
				# print("fitparams",funatlas.fconn[ds].fit_params[ie][resp_i])
				# print("fitparamsdefault",funatlas.fconn[ds].fit_params_default)

				if not np.array_equal(funatlas.fconn[ds].fit_params[ie][resp_i]['params'], funatlas.fconn[ds].fit_params_default['params']) or not funatlas.fconn[ds].fit_params[ie][resp_i]['n_branches'] == funatlas.fconn[ds].fit_params_default['n_branches'] or not np.array_equal(funatlas.fconn[ds].fit_params[ie][resp_i]['n_branch_params'], funatlas.fconn[ds].fit_params_default['n_branch_params']):

					# Generate the fitted stimulus activity
					stim_unc_par = funatlas.fconn[ds].fit_params_unc[ie][stim_j]  # what is the second index
					if stim_unc_par["n_branches"]==0:
						print("skipping "+str(ie))
						continue

					else:

						# y = funatlas.fconn[ds].get_kernel(time, ie, resp_i)
						ker_ec = funatlas.fconn[ds].get_kernel_ec(ie, resp_i)
						stim_ec = pp.ExponentialConvolution.from_dict(stim_unc_par)
						# f = ec.eval(time)

						all_kernels_ec.append(ker_ec)
						all_fitted_stims_ec.append(stim_ec)
						all_dt.append(Dt)

						kernels_by_pair_ec[str(pair_ind)].append(ker_ec)
						fitted_stims_by_pair_ec[str(pair_ind)].append(stim_ec)
						dt_by_pair[str(pair_ind)].append(Dt)

						occ_ind = occ_ind + 1
						# print("occ_ind",occ_ind)


						# lbl = "dataset: " + str(ds) + "stimnum: " + str(ie)
						# ax1.plot(time, y, label=lbl, linewidth=5)
						# ax1.set_xlim(0,)
						# norm_range = (None,shift_vol+int(40./Dt))
						# funatlas.sig[ds].smooth(n=127,i=None,poly=7,mode="sg")
						# y_trace = funatlas.sig[ds].get_segment(i0,i1,shift_vol,normalize=normalize)[:,resp_i]
						# lbl2 = "resp: dataset: " + str(ds) + "stimnum: " + str(ie)
						# lbl3 = "fitted stim: dataset: " + str(ds) + "stimnum: " + str(ie)
						# ax2.plot(time_trace, y_trace, label=lbl2, linewidth=5)
						# ax2.plot(time, f, label=lbl3, linewidth=7)
						# ax2.set_xlim(0,)




			# ax1.legend()
			# ax1.set_xlabel("time (s)")
			# ax1.set_ylabel("Kernel (arb. u.)")
			# ax1.set_title("Kernels of stimulation of "+iid+" and response by " + jid)
			# ax2.legend()
			# ax2.set_xlabel("time (s)")
			# ax2.set_ylabel("G/R (arb. u.)")
			# ax2.set_title("Respnse of "+jid+" by stimulation of " + iid)
			# fig.savefig("/projects/LEIFER/Sophie/comp_Resp_Fig/"+"stim_"+iid+"_resp_"+jid+".png",bbox_inches="tight")
			# fig.clf()
			pair_ind = pair_ind + 1



#time = range(0,30,0.5)
dt = 0.5

all_comp_resp_corr = []


for ker in range(len(all_kernels_ec)):

	for ker2 in range(ker+1, len(all_kernels_ec)):

		if ker != ker2:



			ec1 = all_fitted_stims_ec[ker]
			ec2 = all_fitted_stims_ec[ker2]
			fit_stim1 = ec1.eval(time)
			fit_stim2 = ec2.eval(time)


			kernel1 = all_kernels_ec[ker].eval(time)
			kernel2 = all_kernels_ec[ker2].eval(time)

			response11 = pp.convolution(fit_stim1, kernel1, dt, 8)
			response12 = pp.convolution(fit_stim1, kernel2, dt, 8)
			corr_coef1, _ = stats.pearsonr(response11, response12)

			response21 = pp.convolution(fit_stim2, kernel1, dt, 8)
			response22 = pp.convolution(fit_stim2, kernel2, dt, 8)
			corr_coef2, _ = stats.pearsonr(response21, response22)

			all_comp_resp_corr.append(corr_coef1)
			all_comp_resp_corr.append(corr_coef2)






pair_comp_resp_coeff = []

# print(pair_ind)
for i in range(pair_ind - 1):
	i = i+1
	if len(kernels_by_pair_ec[str(i)]) > 2:
		for k in range(len(kernels_by_pair_ec[str(i)])):

			for k2 in range(k+1, len(kernels_by_pair_ec[str(i)])):

				if k != k2:

					ec1 = fitted_stims_by_pair_ec[str(i)][k]
					ec2 = fitted_stims_by_pair_ec[str(i)][k2]

					fit_stim1 = ec1.eval(time)
					fit_stim2 = ec2.eval(time)

					kernel1 = kernels_by_pair_ec[str(i)][k].eval(time)
					kernel2 = kernels_by_pair_ec[str(i)][k2].eval(time)

					response11 = pp.convolution(fit_stim1, kernel1, dt, 8)
					response12 = pp.convolution(fit_stim1, kernel2, dt, 8)
					corr_coef_pair1, _ = stats.pearsonr(response11, response12)

					if corr_coef_pair1 > 0.95 and plots_make:
						fig1 = plt.figure(1, figsize=(15, 8))
						stim_neu_id = pair_by_pair[str(i)][1]
						resp_neu_id = pair_by_pair[str(i)][0]

						lblfit = "fitted stimulus response of " + stim_neu_id +"corresponding to kernel 1"
						lblker1 = "Kernel 1 between " + stim_neu_id + " and " + resp_neu_id
						lblresp1 = "Response of " + resp_neu_id + " from convolving the fitted stimulus with Kernel 1"
						lblker2 = "Kernel 2 between " + stim_neu_id + " and " + resp_neu_id
						lblresp2 = "Response of " + resp_neu_id + " from convolving the fitted stimulus with Kernel 2"
						plt.plot(time, fit_stim1, label=lblfit, linewidth=5, color='darkred')
						plt.plot(time, kernel1, label=lblker1, linewidth=5, color='lightskyblue', linestyle='dashed')
						plt.plot(time, response11, label=lblresp1, linewidth=5, color='midnightblue')
						plt.plot(time, kernel2, label=lblker2, linewidth=5, color='lightgreen', linestyle='dashed')
						plt.plot(time, response12, label=lblresp2, linewidth=5, color='darkgreen')

						plt.legend()
						plt.xlabel("Time")
						plt.ylabel("Kernel/ Response")
						plt.title("Comparing two responses generated using two kernels between " + stim_neu_id + " and " + resp_neu_id + "with a Correlation Coefficient of " +str(corr_coef_pair1))

						fig1.savefig("/projects/LEIFER/Sophie/comp_Resp_Fig/Comparing_Generated_Responses_High_Corr_"+str(corr_coef_pair1)+"_"+stim_neu_id+"_"+resp_neu_id+".png",
									 bbox_inches="tight")
						fig1.clf()


					elif corr_coef_pair1 < -0.95 and plots_make:
						fig1 = plt.figure(1, figsize=(15, 8))
						stim_neu_id = pair_by_pair[str(i)][1]
						resp_neu_id = pair_by_pair[str(i)][0]

						lblfit = "fitted stimulus response of " + stim_neu_id + "corresponding to kernel 1"
						lblker1 = "Kernel 1 between " + stim_neu_id + " and " + resp_neu_id
						lblresp1 = "Response of " + resp_neu_id + " from convolving the fitted stimulus with Kernel 1"
						lblker2 = "Kernel 2 between " + stim_neu_id + " and " + resp_neu_id
						lblresp2 = "Response of " + resp_neu_id + " from convolving the fitted stimulus with Kernel 2"
						plt.plot(time, fit_stim1, label=lblfit, linewidth=5, color='darkred')
						plt.plot(time, kernel1, label=lblker1, linewidth=5, color='lightskyblue', linestyle='dashed')
						plt.plot(time, response11, label=lblresp1, linewidth=5, color='midnightblue')
						plt.plot(time, kernel2, label=lblker2, linewidth=5, color='lightgreen', linestyle='dashed')
						plt.plot(time, response12, label=lblresp2, linewidth=5, color='darkgreen')

						plt.legend()
						plt.xlabel("Time")
						plt.ylabel("Kernel/ Response")
						plt.title(
							"Comparing two responses generated using two kernels between " + stim_neu_id + " and " + resp_neu_id + "with a Correlation Coefficient of " + str(
								corr_coef_pair1))

						fig1.savefig(
							"/projects/LEIFER/Sophie/comp_Resp_Fig/Comparing_Generated_Responses_Neg_High_Corr_" + str(
								corr_coef_pair1) + "_" + stim_neu_id + "_" + resp_neu_id + ".png",
							bbox_inches="tight")
						fig1.clf()

					elif 0.05 > corr_coef_pair1 > -0.05 and plots_make:

						fig1 = plt.figure(1, figsize=(15, 8))
						stim_neu_id = pair_by_pair[str(i)][1]
						resp_neu_id = pair_by_pair[str(i)][0]

						lblfit = "fitted stimulus response of " + stim_neu_id + "corresponding to kernel 1"
						lblker1 = "Kernel 1 between " + stim_neu_id + " and " + resp_neu_id
						lblresp1 = "Response of " + resp_neu_id + " from convolving the fitted stimulus with Kernel 1"
						lblker2 = "Kernel 2 between " + stim_neu_id + " and " + resp_neu_id
						lblresp2 = "Response of " + resp_neu_id + " from convolving the fitted stimulus with Kernel 2"
						plt.plot(time, fit_stim1, label=lblfit, linewidth=5, color='darkred')
						plt.plot(time, kernel1, label=lblker1, linewidth=5, color='lightskyblue', linestyle='dashed')
						plt.plot(time, response11, label=lblresp1, linewidth=5, color='midnightblue')
						plt.plot(time, kernel2, label=lblker2, linewidth=5, color='lightgreen', linestyle='dashed')
						plt.plot(time, response12, label=lblresp2, linewidth=5, color='darkgreen')

						plt.legend()
						plt.xlabel("Time")
						plt.ylabel("Kernel/ Response")
						plt.title(
							"Comparing two responses generated using two kernels between " + stim_neu_id + " and " + resp_neu_id + "with a Correlation Coefficient of " + str(
								corr_coef_pair1))

						fig1.savefig(
							"/projects/LEIFER/Sophie/comp_Resp_Fig/Comparing_Generated_Responses_Low_Corr_" + str(
								corr_coef_pair1) + "_" + stim_neu_id + "_" + resp_neu_id + ".png",
							bbox_inches="tight")
						fig1.clf()



					response21 = pp.convolution(fit_stim2, kernel1, dt, 8)
					response22 = pp.convolution(fit_stim2, kernel2, dt, 8)
					corr_coef_pair2, _ = stats.pearsonr(response21, response22)

					pair_comp_resp_coeff.append(corr_coef_pair1)
					pair_comp_resp_coeff.append(corr_coef_pair2)



numpairs_tot = pair_ind - 1

num_corr_all = len(all_comp_resp_corr)

num_corr_pairs = len(pair_comp_resp_coeff)

if merge_LR:
	txt_capt = "The distribution of Pearson correlation coefficients between pairs of responses generated by the same stimulus using \
		 kernels taken from all pairs of neurons (Orange, n-ceoff = " + str(
		num_corr_all) + ") \n and taken from the same stimulated and responding neurons (Blue, n-coeff = " \
			   + str(num_corr_pairs) + "). Number of recordings: " + str(nr_of_datasets) + ". Neurons have been bilaterally merged."

else:
	txt_capt = "The distribution of Pearson correlation coefficients between pairs of responses generated by the same stimulus using \
	 kernels taken from all pairs of neurons (Orange, n-ceoff = " + str(num_corr_all) + ") \n and taken from the same stimulated and responding neurons (Blue, n-coeff = " \
		+ str(num_corr_pairs) +"). Number of recordings: " + str(nr_of_datasets) + "."


print(txt_capt)

fname_add = "_merged_" if merge_LR else ""
try:
    np.savetxt("/projects/LEIFER/francesco/funatlas/diagnostics/allcorrelations"+fname_add+"_pair_comp_resp_coeff.txt",pair_comp_resp_coeff)
    np.savetxt("/projects/LEIFER/francesco/funatlas/diagnostics/allcorrelations"+fname_add+"_all_comp_resp_corr.txt",all_comp_resp_corr)
except:
    pass
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations"+fname_add+"_pair_comp_resp_coeff.txt",pair_comp_resp_coeff)
np.savetxt("/projects/LEIFER/francesco/funatlas/figures/paper/figS9/allcorrelations"+fname_add+"_all_comp_resp_corr.txt",all_comp_resp_corr)

nbins = 30
fig2 = plt.figure(1, figsize=(15, 10), dpi = 100)
legend = ('Responses generated \n from the same pair of neurons','All generated responses shuffled')
plt.hist(pair_comp_resp_coeff, bins = 30, alpha = 0.5, density = "True")
plt.hist(all_comp_resp_corr, bins = 30, alpha = 0.5, density = "True")
plt.xlabel("Correlation coefficient", fontsize=40)
plt.ylabel("Density", fontsize=40)
plt.legend(legend, fontsize=30)
plt.xticks(np.arange(-1, 1, step=0.5), fontsize= 40)
plt.yticks(np.arange(0, 1.5, step=0.5), fontsize= 40)
# plt.title("Responses From the Same Pair of Stimulated and Responding Neurons are More Stereotyped than Random", fontsize=16)
# Density of Correlations Between Responses with the Same Pair of Neurons or Different Pairs of Neurons that have been Generated Using the Same Stimulus and Respective Kernels'
# fig2.text(.5, 0, txt_capt, ha='center')
# plt.show()
'''if merge_LR:
	fig2.savefig("/projects/LEIFER/Sophie/comp_Resp_Fig/allcorrelations_merged.png",bbox_inches="tight")
	fig2.savefig("/projects/LEIFER/francesco/funatlas/diagnostics/allcorrelations_merged.png", bbox_inches="tight")
else:
	fig2.savefig("/projects/LEIFER/Sophie/comp_Resp_Fig/allcorrelationsway_notmerged.png", bbox_inches="tight")
	fig2.savefig("/projects/LEIFER/francesco/funatlas/diagnostics/allcorrelationsway_notmerged.png", bbox_inches="tight")'''
fig2.clf()












