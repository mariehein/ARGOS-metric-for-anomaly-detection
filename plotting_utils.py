import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.cm as col
import matplotlib as mpl
import warnings
import json
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D

c_RWTH = {'b': '#00549F',  # blue
          'lb': '#8EBAE5', # light blue
          'dr': '#A11035', # dark red
          'g': '#57AB27',  # green
          't': '#006165',  # teal / petrol
          'o': '#F6A800',  # orange
          'lg': '#BDCD00', # light green
          'gr': '#646567', # gray
          'v': '#612158',  # violett
          'r': '#CC071E',  # red
          'tq': '#0098A1', # turquoise
          'p': '#7A6FAC'}  # purple

#classifier_colors = {"NN": c_RWTH["b"], "HGB": c_RWTH["v"], "AdaBoost": c_RWTH["o"]}#

#classifier_colors = {"NN": "dodgerblue", "HGB": "red", "AdaBoost": "orange"}
#classifier_colors = {"NN": '#1a80bb', "HGB": '#a00000', "AdaBoost": '#ea801c'}
#classifier_colors = {"NN": '#40ad5a', "HGB": '#3c93c2', "AdaBoost": '#F28522'}
#classifier_colors = {"NN": 'darkorange', "HGB": 'teal', "AdaBoost": 'mediumvioletred'}
#classifier_colors = {"NN": '#0072B2', "HGB": '#56b4e9', "AdaBoost": '#e69f00'}
classifier_colors = {"NN": "blue", "HGB": "red", "AdaBoost": "orange"}

#metric_colors = {"val_SIC": c_RWTH["r"], "val_loss": c_RWTH["tq"], "max_SIC": c_RWTH["p"]}
metric_colors = {"val_SIC": "dodgerblue", "val_loss": "mediumvioletred", "max_SIC": "darkorange"}


metric_names = {"val_SIC": "ARGOS", "val_loss": "BCE", "max_SIC": "Max SIC"}
metric_NN_files_names= {"val_SIC": "val_sic", "val_loss": "val_loss", "max_SIC": "max_sic"}
metric_maximize= {"val_SIC": 1, "val_loss": -1, "max_SIC": 1}
mode_name = {"IAD": "IAD", "cathode": "CATHODE", "cwola": "CWoLa"}
default_metric = "val_sic"
default_classifier = "NN"
max_SIC_lims=17

feature_set_names = {"baseline": "Baseline features", "extended1": "Extended set 1", "extended2": "Extended set 2", "extended3": "Extended set 3",  "DeltaR": r"Baseline with $\Delta R$", "useless": "Uninformative features"}
feature_set_colors = {"baseline": "grey", "extended1": "red", "extended2": "orange", "extended3": "dodgerblue",  "DeltaR": "purple", "useless": "blueviolet"}

max_err=0.2

S = 20000
B = 312858

plt.rcParams['pgf.rcfonts'] = False
#plt.rcParams['font.serif'] = []
#plt.rcParams['font.family']="serif"
#pl.rcParams['mathtext.fontset']="cm"
plt.rcParams['figure.figsize'] = 7, 5
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = True

points = np.array([])
index = -1

def read_ROC_SIC_1D(path, points, folder, N_runs=10, start_runs=0, return_full_array=False):
	fpr=np.load(folder+"fpr_"+path+".npy")[start_runs:start_runs+N_runs]
	#print(fpr.shape)
	tpr=np.load(folder+"tpr_"+path+".npy")[start_runs:start_runs+N_runs]
	ROC_values = np.zeros((len(fpr),len(points)))
	SIC_values = np.zeros((len(fpr),len(points)+2))
	for j in range(len(ROC_values)):
		inds = np.nonzero(tpr[j])[0]
		t = tpr[j, inds]
		f = fpr[j, inds]
		ROC_values[j] = interp1d(t, 1/f)(points)
		SIC_values[j,:-2] = interp1d(t, t/np.sqrt(f))(points)
		SIC_values[j,-1] = np.nanmax(np.nan_to_num(t/np.sqrt(f), posinf=0),where= f>1/312858/max_err**2,initial=0)
		SIC_values[j,-2] = np.nanmax(np.nan_to_num(t/np.sqrt(f),posinf=0))
	if return_full_array:
		return SIC_values[:,-1]
	else:
		return np.median(SIC_values,axis=0), np.percentile(SIC_values, 16, axis=0), np.percentile(SIC_values, 84, axis=0)

def read_metric(path, metric, NN, start_runs=0, N_runs=10, max=1):
	metric_values = np.zeros(N_runs-start_runs)
	for i in range(start_runs, N_runs):
		if NN:
			metric_values[i] = json.load(open(path+"run"+str(i)+"/"+metric+"_averaged.json"))[metric]
		else: 
			metric_values[i] = np.load(path+"run"+str(i)+"/"+metric+".npy")
	return metric_values	

def plot_end_1D(fig, ax, sig, name, small=True, ylim=max_SIC_lims, title=None, plotting_directory="plots/", loc="lower right"): 
	ax.legend(loc=loc)
	ax.set_ylabel(r"$\max\ \epsilon_S/\sqrt{\epsilon_B}$")
	ax.set_xlabel(r"$N_{sig}$")
	ax.set_xticks(sig)
	ax.set_xlim(min(sig), max(sig))
	ax.set_ylim(0, ylim)
	
	if title is not None:
		ymin, ymax = plt.ylim()
		xmin, xmax = plt.xlim()
		a = 0.03
		plt.text(xmin + a * (xmax-xmin), ymin + (1-a) *(ymax-ymin) , title, size=plt.rcParams['axes.labelsize'], color='black', horizontalalignment='left', verticalalignment='top')
	
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_"+name+".pdf")
	plt.show()

def plot_end_1D_multiple(ax, sig, ylim=max_SIC_lims, ylabel=None, ylims=None, title=None, loc="lower right", legend=True): 
	if legend:
		ax.legend(loc=loc)
	if ylabel is None:
		ax.set_ylabel(r"$\max\ \epsilon_S/\sqrt{\epsilon_B}$")
	else: 
		ax.set_ylabel(ylabel)
	ax.set_xlabel(r"$N_{sig}$")
	ax.set_xticks(sig)
	ax.set_xlim(min(sig), max(sig))
	if ylims is None:
		ax.set_ylim(0, ylim)
	else:
		ax.set_ylim(*ylims)
	
	if title is not None:
		ymin, ymax = ax.get_ylim()
		xmin, xmax = ax.get_xlim()
		a = 0.03
		ax.text(xmin + a * (xmax-xmin), ymin + (1-a) *(ymax-ymin) , title, size=plt.rcParams['axes.labelsize'], color='black', horizontalalignment='left', verticalalignment='top')

def plot_sic(ax, sic, sic_low, sic_upp, sig, color, label, normed=False, linestyle="solid"):
	if normed:
		n = np.max(sic)
		ax.plot(sig, sic/n, color=color,label=label, linestyle=linestyle)
		ax.fill_between(sig, sic_low/n, sic_upp/n, alpha=0.2, facecolor=color)
		return
	ax.plot(sig, sic, color=color,label=label, linestyle=linestyle, marker='o')
	ax.fill_between(sig, sic_low, sic_upp, alpha=0.2, facecolor=color)

def read_max_SIC(path, start_runs=0, N_runs=10):
	maxSICs = np.zeros(N_runs)
	for run in range(start_runs, start_runs+N_runs):
		maxSICs[run-start_runs] = np.max(np.load(path+"run"+str(run)+"/CLSF_max_SIC.npy"))
	return np.median(maxSICs), np.percentile(maxSICs, 16, axis=0), np.percentile(maxSICs, 84, axis=0)	

def plot_default_runs(mode, signals, signals_plot, gen_direc, plotting_directory, rotated=False, classifiers=["NN", "HGB", "AdaBoost"]):
	sic = np.zeros((len(signals),len(points)+2))
	sic_low = np.zeros((len(signals),len(points)+2))
	sic_upp = np.zeros((len(signals),len(points)+2))

    #fig, ax = plt.subplots(figsize=(9,6))
	fig, ax = plt.subplots()
	for c in classifiers: 
		if c =="NN":
			roc_name = default_metric   
		else: 
			roc_name = "BDT"
		if rotated:
			folder = gen_direc+c+"/default_hp/"+mode+"_rotated/"
			title = mode_name[mode]+", rotated"
		else:
			folder = gen_direc+c+"/default_hp/"+mode+"/"
			title = mode_name[mode]
		for j,s in enumerate(signals):
			sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
		plot_sic(ax, sic[:,index], sic_low[:,index], sic_upp[:,index], signals, classifier_colors[c], c)
	plot_end_1D(fig, ax, signals_plot, mode+"_default", ylim=max_SIC_lims, title=title, plotting_directory=plotting_directory)

def plot_epoch_selection(mode, signals, signals_plot, gen_direc, plotting_directory, rotated=False, signal_specific=500, run=0, loc="lower right", half=False, metrics=["max_SIC", "val_loss", "val_SIC"], classifier="NN"):
	sic = np.zeros((len(signals),len(points)+2))
	sic_low = np.zeros((len(signals),len(points)+2))
	sic_upp = np.zeros((len(signals),len(points)+2))

	if rotated:
		folder = gen_direc+classifier+"/default_hp/"+mode+"_rotated/"
		name=mode+"_rotated"
	else:
		folder = gen_direc+classifier+"/default_hp/"+mode+"/"
		name=mode

	fig, ax = plt.subplots(1,len(metrics), figsize=(15,5))
	for i,m in enumerate(metrics):
		metric = np.load(folder+"Nsig_"+str(signal_specific)+"/run"+str(run)+"/CLSF_"+m+".npy")
		ax[i].plot(np.arange(1,len(metric)+1),metric, color=metric_colors[m], label=metric_names[m])
		ax[i].set_ylabel(metric_names[m])
		ax[i].set_xlabel("Epoch")
		if m=="max_SIC":
			ax[i].set_ylim(0,max_SIC_lims)

		title=metric_names[m]
		ymin, ymax = ax[i].get_ylim()
		xmin, xmax = ax[i].get_xlim()
		a = 0.03
		ax[i].text(xmin + a * (xmax-xmin), ymin + (1-a) *(ymax-ymin) , title, size=plt.rcParams['axes.labelsize'], color='black', horizontalalignment='left', verticalalignment='top')
	fig.tight_layout()
	fig.savefig(plotting_directory+"metric_tracking_"+name+"_"+str(signal_specific)+".pdf")

	"""
	title = mode_name[mode]+r" at $N_{sig}=$"+str(signal_specific)
	ymin, ymax = plt.ylim()
	xmin, xmax = plt.xlim()
	a = 0.03
	plt.text(xmin + a * (xmax-xmin), ymin + (1-a) *(ymax-ymin) , title, size=plt.rcParams['axes.labelsize'], color='black', horizontalalignment='left', verticalalignment='top')
	
	fig.tight_layout()
	fig.savefig(plotting_directory+"metric_tracking_"+name+"_"+str(signal_specific)+".pdf")
	plt.show()"""

	fig2, ax2 = plt.subplots()
	for i,m in enumerate(metrics): 
		roc_name = metric_NN_files_names[m]
		for j,s in enumerate(signals):
			sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
		plot_sic(ax2, sic[:,index], sic_low[:,index], sic_upp[:,index], signals, metric_colors[m], "Epochs seclected on "+metric_names[m])
	
	plot_end_1D(fig2, ax2, signals_plot, loc=loc, ylim=max_SIC_lims, plotting_directory=plotting_directory, name=name+"_epoch_selection")
	plt.show()


def plot_optimized_runs(mode, signals, signals_plot, gen_direc, plotting_directory, plot_default=True, rotated=False, loc="lower right", half=False, metrics=["max_SIC", "val_loss", "val_SIC"], classifiers=["NN", "HGB", "AdaBoost"]):
	sic = np.zeros((len(signals),len(points)+2))
	sic_low = np.zeros((len(signals),len(points)+2))
	sic_upp = np.zeros((len(signals),len(points)+2))

	#figsize=(9,6)
    	
	fig, ax = plt.subplots(1,len(classifiers), figsize=(15,5))
	if len(classifiers)==1:
		ax = [ax]
	for i,c in enumerate(classifiers): 
		if c =="NN":
			roc_name = default_metric
		else: 
			roc_name = "BDT"
		if c=="HGB":
			legend=True
		else: 
			legend=False
		if rotated:
			folder = gen_direc+c+"/default_hp/"+mode+"_rotated/"
		else:
			folder = gen_direc+c+"/default_hp/"+mode+"/"
		if plot_default:
			for j,s in enumerate(signals):
				sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
			plot_sic(ax[i], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, "black", "Default")

		for m in metrics:
			if c =="NN":
				roc_name = metric_NN_files_names[m]
			else: 
				roc_name = "BDT"
			if rotated:
				folder = gen_direc+c+"/optimized_hp/"+mode+"_rotated/"+m+"/"
				if half: 
					folder = gen_direc+c+"/optimized_hp/"+mode+"_rotated_half/"+m+"/"
				name= mode+"_rotated"
			else:
				folder = gen_direc+c+"/optimized_hp/"+mode+"/"+m+"/"
				if half: 
					folder = gen_direc+c+"/optimized_hp/"+mode+"_half/"+m+"/"
				name=mode
			for j,s in enumerate(signals):
				#print(s)
				sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
			plot_sic(ax[i], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, metric_colors[m], "Optimized on "+metric_names[m])
		plot_end_1D_multiple(ax[i], signals_plot,title=c, legend=legend, loc=loc)
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_" +name+"_optimized.pdf")
	plt.show()

def plot_selection(mode, signals, signals_plot, gen_direc, plotting_directory, second_axis=True, loc="lower right", plot_default=True, rotated=False, metric="val_SIC", metric_opt=None, classifiers=["NN", "HGB", "AdaBoost"]):    
	fig, ax = plt.subplots(1,3, figsize=(15,5))

	if metric_opt is None: 
		metric_opt = metric

	if plot_default:
		sic = np.zeros((len(signals),len(points)+2))
		sic_low = np.zeros((len(signals),len(points)+2))
		sic_upp = np.zeros((len(signals),len(points)+2))

		if default_classifier=="NN":
			roc_name = metric_NN_files_names[metric]
		else:
			roc_name = "BDT"
		if rotated:
			folder = gen_direc+default_classifier+"/default_hp/"+mode+"_rotated/"
		else:
			folder = gen_direc+default_classifier+"/default_hp/"+mode+"/"
		for j,s in enumerate(signals):
			sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
		plot_sic(ax[2], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, "grey", "Default", linestyle="dashed")

	#if second_axis: 
	#	ax2 = ax[0].twinx()
	#	ax2.set_ylabel(metric_names[metric])

	sic = np.zeros((len(classifiers), len(signals),10))
	sic_half = np.zeros((len(classifiers), len(signals),10))
	metric_values = np.zeros((len(classifiers), len(signals), 10))
	for i,c in enumerate(classifiers): 
		if c =="NN":
			roc_name = metric_NN_files_names[metric]
			is_NN = True
		else: 
			roc_name = "BDT"
			is_NN = False
		if rotated:
			folder = gen_direc+c+"/optimized_hp/"+mode+"_rotated/"+metric_opt+"/"
			folder_half = gen_direc+c+"/optimized_hp/"+mode+"_rotated_half/"+metric_opt+"/"
			name = mode+"_rotated"
		else:
			folder = gen_direc+c+"/optimized_hp/"+mode+"/"+metric_opt+"/"
			folder_half = gen_direc+c+"/optimized_hp/"+mode+"_half/"+metric_opt+"/"
			name=mode

		for j,s in enumerate(signals):
			sic[i,j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/", return_full_array=True)
			sic_half[i,j] = read_ROC_SIC_1D(roc_name, points, folder_half + "Nsig_"+str(s)+"/", return_full_array=True)
			metric_values[i,j] = read_metric(folder_half+ "Nsig_"+str(s)+"/", metric, is_NN)
		plot_sic(ax[2], np.median(sic[i], axis=-1), np.percentile(sic[i], 16, axis=-1), np.percentile(sic[i], 84, axis=-1), signals, classifier_colors[c], c)
		plot_sic(ax[0], np.median(metric_values[i], axis=-1), np.percentile(metric_values[i], 16, axis=-1), np.percentile(metric_values[i], 84, axis=-1), signals, classifier_colors[c], c)
		plot_sic(ax[1], np.median(sic_half[i], axis=-1), np.percentile(sic_half[i], 16, axis=-1), np.percentile(sic_half[i], 84, axis=-1), signals, classifier_colors[c], c)
		
		#if second_axis:
		#	ax2.plot(signals, np.median(metric_values[i], axis=-1), color=classifier_colors[c], linestyle="dotted")

	select = np.argmax(metric_maximize[metric]*metric_values, axis=0)
	sic_full_select = np.array([[sic[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
	sic_half_select = np.array([[sic_half[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
	metric_select = np.array([[metric_values[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
		
	plot_sic(ax[2], np.median(sic_full_select, axis=-1), np.percentile(sic_full_select, 16, axis=-1), np.percentile(sic_full_select, 84, axis=-1), signals, "black", "Selected")
	plot_sic(ax[0], np.median(metric_select, axis=-1), np.percentile(metric_select, 16, axis=-1), np.percentile(metric_select, 84, axis=-1), signals, "black", "Selected")
	plot_sic(ax[1], np.median(sic_half_select, axis=-1), np.percentile(sic_half_select, 16, axis=-1), np.percentile(sic_half_select, 84, axis=-1), signals, "black", "Selected")

	plot_end_1D_multiple(ax[1], signals_plot,title="Half statistics", legend=False)
	metmin = np.min(np.percentile(metric_values, 16, axis=-1))
	metmax = np.max(np.percentile(metric_values, 84, axis=-1))
	plot_end_1D_multiple(ax[0], signals_plot,title="Metrics", ylabel=metric_names[metric], legend=False, ylims=(metmin-(metmax-metmin)*0.05, metmax+(metmax-metmin)*0.05))
	plot_end_1D_multiple(ax[2], signals_plot,title="Full statistics", legend=True, loc=loc)
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_" + name+"_"+metric+"_selected.pdf")
	plt.show()

def plot_feature_selection(mode, signals, signals_plot, gen_direc, plotting_directory, ylim=None, second_axis=True, loc="lower right", plot_default=True, rotated=False, metric="val_SIC", feature_sets=["baseline", "extended1", "extended2", "extended3", "DeltaR"], classifier="HGB"):    
	fig, ax = plt.subplots(1,3, figsize=(15,5))

	if plot_default:
		sic = np.zeros((len(signals),len(points)+2))
		sic_low = np.zeros((len(signals),len(points)+2))
		sic_upp = np.zeros((len(signals),len(points)+2))

		if default_classifier=="NN":
			roc_name = metric_NN_files_names[metric]
		else:
			roc_name = "BDT"
		if rotated:
			folder = gen_direc+default_classifier+"/default_hp/"+mode+"_rotated/"
		else:
			folder = gen_direc+default_classifier+"/default_hp/"+mode+"/"
		for j,s in enumerate(signals):
			sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
		plot_sic(ax[2], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, "grey", "Default", linestyle="dashed")

	#if second_axis: 
	#	ax2 = ax[0].twinx()
	#	ax2.set_ylabel(metric_names[metric])

	sic = np.zeros((len(feature_sets), len(signals),10))
	sic_half = np.zeros((len(feature_sets), len(signals),10))
	metric_values = np.zeros((len(feature_sets), len(signals), 10))
	for i,f in enumerate(feature_sets): 
		if f=="baseline":
			feature_set_file = ""
		else: 
			feature_set_file = "_"+f
		if classifier =="NN":
			roc_name = metric_NN_files_names[metric]
			is_NN = True
		else: 
			roc_name = "BDT"
			is_NN = False
		if rotated:
			folder = gen_direc+classifier+"/default_hp/"+mode+"_rotated"+feature_set_file+"/"
			folder_half = gen_direc+classifier+"/default_hp/"+mode+"_rotated_half"+feature_set_file+"/"
			name = mode+"_rotated"
		else:
			folder = gen_direc+classifier+"/default_hp/"+mode+feature_set_file+"/"
			folder_half = gen_direc+classifier+"/default_hp/"+mode+"_half"+feature_set_file+"/"
			name=mode

		for j,s in enumerate(signals):
			sic[i,j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/", return_full_array=True)
			sic_half[i,j] = read_ROC_SIC_1D(roc_name, points, folder_half + "Nsig_"+str(s)+"/", return_full_array=True)
			metric_values[i,j] = read_metric(folder_half+ "Nsig_"+str(s)+"/", metric, is_NN)
		plot_sic(ax[2], np.median(sic[i], axis=-1), np.percentile(sic[i], 16, axis=-1), np.percentile(sic[i], 84, axis=-1), signals, feature_set_colors[f], feature_set_names[f])
		plot_sic(ax[0], np.median(metric_values[i], axis=-1), np.percentile(metric_values[i], 16, axis=-1), np.percentile(metric_values[i], 84, axis=-1), signals, feature_set_colors[f], feature_set_names[f])
		plot_sic(ax[1], np.median(sic_half[i], axis=-1), np.percentile(sic_half[i], 16, axis=-1), np.percentile(sic_half[i], 84, axis=-1), signals, feature_set_colors[f], feature_set_names[f])
		

	select = np.argmax(metric_maximize[metric]*metric_values, axis=0)
	sic_full_select = np.array([[sic[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
	sic_half_select = np.array([[sic_half[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
	metric_select = np.array([[metric_values[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
		
	plot_sic(ax[2], np.median(sic_full_select, axis=-1), np.percentile(sic_full_select, 16, axis=-1), np.percentile(sic_full_select, 84, axis=-1), signals, "black", "Selected")
	plot_sic(ax[0], np.median(metric_select, axis=-1), np.percentile(metric_select, 16, axis=-1), np.percentile(metric_select, 84, axis=-1), signals, "black", "Selected")
	plot_sic(ax[1], np.median(sic_half_select, axis=-1), np.percentile(sic_half_select, 16, axis=-1), np.percentile(sic_half_select, 84, axis=-1), signals, "black", "Selected")

	if ylim is None: 
		ylim=max_SIC_lims

	plot_end_1D_multiple(ax[1], signals_plot, ylim=ylim,title="Half statistics", legend=False)
	metmin = np.min(np.percentile(metric_values, 16, axis=-1))
	metmax = np.max(np.percentile(metric_values, 84, axis=-1))
	plot_end_1D_multiple(ax[0], signals_plot,title="Metrics", ylabel=metric_names[metric], legend=False, ylims=(metmin-(metmax-metmin)*0.05, metmax+(metmax-metmin)*0.05))
	plot_end_1D_multiple(ax[2], signals_plot, ylim=ylim,title="Full statistics", legend=True, loc=loc)
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_" + name+"_"+metric+"_feature_selected.pdf")
	plt.show()

def plot_rotation_selection(mode, signals, signals_plot, gen_direc, plotting_directory, second_axis=True, loc="lower right", plot_default=False, rotated=False, metric="val_SIC", classifier="HGB"):    
	fig, ax = plt.subplots(1,3, figsize=(15,5))

	rotation_states = [False, True]

	if plot_default:
		sic = np.zeros((len(signals),len(points)+2))
		sic_low = np.zeros((len(signals),len(points)+2))
		sic_upp = np.zeros((len(signals),len(points)+2))

		if default_classifier=="NN":
			roc_name = metric_NN_files_names[metric]
		else:
			roc_name = "BDT"
		if rotated:
			folder = gen_direc+default_classifier+"/default_hp/"+mode+"_rotated/"
		else:
			folder = gen_direc+default_classifier+"/default_hp/"+mode+"/"
		for j,s in enumerate(signals):
			sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
		plot_sic(ax[2], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, "grey", "Default", linestyle="dashed")

	#if second_axis: 
	#	ax2 = ax[0].twinx()
	#	ax2.set_ylabel(metric_names[metric])
	name = mode

	sic = np.zeros((len(rotation_states), len(signals),10))
	sic_half = np.zeros((len(rotation_states), len(signals),10))
	metric_values = np.zeros((len(rotation_states), len(signals), 10))
	for i, rotated in enumerate(rotation_states): 
		if classifier =="NN":
			roc_name = metric_NN_files_names[metric]
			is_NN = True
		else: 
			roc_name = "BDT"
			is_NN = False
		if rotated:
			folder = gen_direc+classifier+"/default_hp/"+mode+"_rotated/"
			folder_half = gen_direc+classifier+"/default_hp/"+mode+"_rotated_half/"
			label = "Rotated"
			f = "useless"
		else:
			folder = gen_direc+classifier+"/default_hp/"+mode+"/"
			folder_half = gen_direc+classifier+"/default_hp/"+mode+"_half/"
			label = "Standard"
			f= None

		for j,s in enumerate(signals):
			sic[i,j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/", return_full_array=True)
			sic_half[i,j] = read_ROC_SIC_1D(roc_name, points, folder_half + "Nsig_"+str(s)+"/", return_full_array=True)
			metric_values[i,j] = read_metric(folder_half+ "Nsig_"+str(s)+"/", metric, is_NN)
		plot_sic(ax[2], np.median(sic[i], axis=-1), np.percentile(sic[i], 16, axis=-1), np.percentile(sic[i], 84, axis=-1), signals, feature_set_colors[f], label)
		plot_sic(ax[0], np.median(metric_values[i], axis=-1), np.percentile(metric_values[i], 16, axis=-1), np.percentile(metric_values[i], 84, axis=-1), signals, feature_set_colors[f], label)
		plot_sic(ax[1], np.median(sic_half[i], axis=-1), np.percentile(sic_half[i], 16, axis=-1), np.percentile(sic_half[i], 84, axis=-1), signals, feature_set_colors[f], label)
		

	select = np.argmax(metric_maximize[metric]*metric_values, axis=0)
	sic_full_select = np.array([[sic[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
	sic_half_select = np.array([[sic_half[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
	metric_select = np.array([[metric_values[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
		
	plot_sic(ax[2], np.median(sic_full_select, axis=-1), np.percentile(sic_full_select, 16, axis=-1), np.percentile(sic_full_select, 84, axis=-1), signals, "black", "Selected")
	plot_sic(ax[0], np.median(metric_select, axis=-1), np.percentile(metric_select, 16, axis=-1), np.percentile(metric_select, 84, axis=-1), signals, "black", "Selected")
	plot_sic(ax[1], np.median(sic_half_select, axis=-1), np.percentile(sic_half_select, 16, axis=-1), np.percentile(sic_half_select, 84, axis=-1), signals, "black", "Selected")

	plot_end_1D_multiple(ax[1], signals_plot,title="Half statistics", legend=False)
	metmin = np.min(np.percentile(metric_values, 16, axis=-1))
	metmax = np.max(np.percentile(metric_values, 84, axis=-1))
	plot_end_1D_multiple(ax[0], signals_plot,title="Metrics", ylabel=metric_names[metric], legend=False, ylims=(metmin-(metmax-metmin)*0.05, metmax+(metmax-metmin)*0.05))
	plot_end_1D_multiple(ax[2], signals_plot,title="Full statistics", legend=True, loc=loc)
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_" + name+"_"+metric+"_rotation_selected.pdf")
	plt.show()