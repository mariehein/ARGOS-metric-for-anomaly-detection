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

classifier_colors = {"NN": "blue", "HGB": "red", "AdaBoost": "orange"}
metric_colors = {"val_SIC": "dodgerblue", "val_loss": "mediumvioletred", "max_SIC": "darkorange"}

metric_names = {"val_SIC": "ARGOS", "val_loss": "BCE", "max_SIC": "Max SIC"}
metric_NN_files_names= {"val_SIC": "val_sic", "val_loss": "val_loss", "max_SIC": "max_sic"}
metric_maximize= {"val_SIC": 1, "val_loss": -1, "max_SIC": 1}
mode_name = {"IAD": "IAD", "cathode": "CATHODE", "cwola": "CWoLa Hunting"}
default_metric = "val_sic"
default_classifier = "NN"
max_SIC_lims=17

feature_set_names = {"baseline": "Baseline", "extended1": "Extended 1", "extended2": "Extended 2", "extended3": "Extended 3",  "DeltaR": r"Baseline + $\Delta R$", "useless": "Uninformative features"}
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
	'''
	Loads saved TPR and FPR arrays and returns array with SIC values as median and percentile error bands if return_full_array 
	otherwise as full array
	'''
	fpr=np.load(folder+"fpr_"+path+".npy")[start_runs:start_runs+N_runs]
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

def read_metric(path, metric, NN, start_runs=0, N_runs=10):
	'''
	Reads in metric values saved during runs
	'''
	metric_values = np.zeros(N_runs-start_runs)
	for i in range(start_runs, N_runs):
		if NN:
			metric_values[i] = json.load(open(path+"run"+str(i)+"/"+metric+"_averaged.json"))[metric]
		else: 
			metric_values[i] = np.load(path+"run"+str(i)+"/"+metric+".npy")
	return metric_values	

def ax_title(ax, title):
	'''
	Plots the plot title into the top left corner of the subplot
	'''
	ymin, ymax = ax.get_ylim()
	xmin, xmax = ax.get_xlim()
	a = 0.03
	ax.text(xmin + a * (xmax-xmin), ymin + (1-a) *(ymax-ymin) , title, size=plt.rcParams['axes.labelsize'], color='black', horizontalalignment='left', verticalalignment='top')

def plot_end_1D(fig, ax, sig, name, ylim=max_SIC_lims, title=None, plotting_directory="plots/", loc="lower right"): 
	''' 
	Plot design function for singular plot
	'''
	ax.legend(loc=loc)
	ax.set_ylabel("Max SIC")#(r"$\max\ \epsilon_S/\sqrt{\epsilon_B}$")
	ax.set_xlabel(r"$N_{sig}$")
	ax.set_xticks(sig)
	ax.set_xlim(min(sig), max(sig))
	ax.set_ylim(0, ylim)
	if title is not None:
		ax_title(ax, title)
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_"+name+".pdf")
	plt.show()

def plot_end_1D_multiple(ax, sig, ylim=max_SIC_lims, ylabel=None, ylims=None, title=None, loc="lower right", legend=True): 
	''' 
	Plot design function for subplot
	'''
	if legend:
		ax.legend(loc=loc)
	if ylabel is None:
		ylabel="Max SIC"
	ax.set_ylabel(ylabel)
	ax.set_xlabel(r"$N_{sig}$")
	ax.set_xticks(sig)
	ax.set_xlim(min(sig), max(sig))
	if ylims is None:
		ylims=(0,ylim)
	ax.set_ylim(*ylims)
	if title is not None:
		ax_title(ax, title)

def plot_sic(ax, sic, sic_low, sic_upp, sig, color, label, linestyle="solid", alpha=0.2, alpha_line=1.):
	ax.plot(sig, sic, color=color,label=label, linestyle=linestyle, marker='o', alpha=alpha_line)
	ax.fill_between(sig, sic_low, sic_upp, alpha=alpha, facecolor=color)

def plot_epoch_selection(ax2, mode, signals, signals_plot, gen_direc, plotting_directory, legend=False, rotated=False, signal_specific=500, run=0, loc="lower right", half=False, metrics=["max_SIC", "val_loss", "val_SIC"], classifier="NN"):
	'''
	Plots the epoch selection for the given parameters. Also plots example metric curves for each mode. 
	'''
	sic = np.zeros((len(signals),len(points)+2))
	sic_low = np.zeros((len(signals),len(points)+2))
	sic_upp = np.zeros((len(signals),len(points)+2))

	suffix = "_rotated" if rotated else ""
	folder = f"{gen_direc}{classifier}/default_hp/{mode}{suffix}/"
	name = f"{mode}{suffix}"

	fig, ax = plt.subplots(1,len(metrics), figsize=(15,5))
	for i,m in enumerate(metrics):
		metric = np.load(folder+"Nsig_"+str(signal_specific)+"/run"+str(run)+"/CLSF_"+m+".npy")
		ax[i].plot(np.arange(1,len(metric)+1),metric, color=metric_colors[m], label=metric_names[m])
		ax[i].set_ylabel(metric_names[m])
		ax[i].set_xlabel("Epoch")
		if m=="max_SIC":
			ax[i].set_ylim(0,max_SIC_lims)
		ax_title(ax[i], metric_names[m])
	fig.tight_layout()
	fig.savefig(plotting_directory+"metric_tracking_"+name+"_"+str(signal_specific)+".pdf")

	for i,m in enumerate(metrics): 
		roc_name = metric_NN_files_names[m]
		for j,s in enumerate(signals):
			sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
		plot_sic(ax2, sic[:,index], sic_low[:,index], sic_upp[:,index], signals, metric_colors[m], metric_names[m]+" selection")
	
	plot_end_1D_multiple(ax2, signals_plot,  title=mode_name[mode], loc=loc, ylim=max_SIC_lims, legend=legend)


def plot_epoch_selection_multiple(signals, signals_plot, gen_direc, plotting_directory, modes = ["IAD", "cwola", "cathode"], rotated=False, 
								  signal_specific=[400, 800, 500], run=[5,2,0], loc="lower right", half=False, legend=True,
								  metrics=["max_SIC", "val_loss", "val_SIC"], classifier="NN", mode_legend="IAD", legend_loc="lower right"):
	'''
	For each mode (IAD, CWoLa Hunting, CATHODE) plots the epoch selection using plot_epoch_selection
	'''
	fig, ax = plt.subplots(1,len(modes), figsize=(15,5))
	for i,mode in enumerate(modes): 
		legend = (mode == mode_legend and legend)
		plot_epoch_selection(ax[i], mode, signals, signals_plot, gen_direc, plotting_directory, legend=legend, loc=legend_loc, 
								signal_specific=signal_specific[i], run=run[i], metrics=metrics, classifier=classifier)
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_epoch_selection_all.pdf")


def plot_optimized_runs(mode, signals, signals_plot, gen_direc, plotting_directory, plot_default=True, rotated=False, loc="lower right", half=False, metrics=["max_SIC", "val_loss", "val_SIC"], classifiers=["NN", "HGB", "AdaBoost"], legend_param = True):
	'''
	Plots hyperparameter optimization results
	'''
	sic = np.zeros((len(signals),len(points)+2))
	sic_low = np.zeros((len(signals),len(points)+2))
	sic_upp = np.zeros((len(signals),len(points)+2))

	fig, ax = plt.subplots(1,len(classifiers), figsize=(15,5))
	ax = np.atleast_1d(ax)

	def folder_path(base, clf, mode, rotated=False, optimized=False, metric=None, half=False):
		suffix = "_rotated" if rotated else ""
		half_suf = "_half" if half else ""
		opt = "optimized_hp" if optimized else "default_hp"
		metric_part = f"{metric}/" if metric else ""
		return f"{base}{clf}/{opt}/{mode}{suffix}{half_suf}/{metric_part}"

	for i,c in enumerate(classifiers): 
		legend = (c=="HGB" and legend_param)
		roc_name = default_metric if c=="NN" else "BDT"

		if plot_default:
			folder = folder_path(gen_direc, c, mode, rotated)
			for j,s in enumerate(signals):
				sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
			plot_sic(ax[i], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, "black", "Default")

		for m in metrics:
			roc_name = metric_NN_files_names[m] if c == "NN" else "BDT"
			folder = folder_path(gen_direc, c, mode, rotated, optimized=True, metric=m, half=half)
			for j,s in enumerate(signals):
				sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
			plot_sic(ax[i], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, metric_colors[m], metric_names[m]+" optimization")
		plot_end_1D_multiple(ax[i], signals_plot,title=mode_name[mode]+": "+c, legend=legend, loc=loc)
	
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_"+mode+"{'_rotated' if rotated else ''}_optimized.pdf")
	plt.show()


def plot_selection_single(mode, signals, signals_plot, gen_direc, plotting_directory, loc="lower right", plot_default=True, 
						  rotated=False, metric="val_SIC", metric_opt=None, classifiers=["NN", "HGB", "AdaBoost"]):    
	'''
	Plot more detailed version of singular metric architecture selection
	'''
	
	fig, ax = plt.subplots(1,3, figsize=(15,5))
	metric_opt = metric_opt or metric
	
	suffix = "_rotated" if rotated else ""
	def folder_path(base, clf, mode, half=False, default=False, metric=None):
		hp_type = "default_hp" if default else "optimized_hp"
		half_suffix = "_half" if half else ""
		metric_path = f"{metric}/" if metric and not default else ""
		return f"{base}{clf}/{hp_type}/{mode}{suffix}{half_suffix}/{metric_path}"

	if plot_default:
		sic = np.zeros((len(signals),len(points)+2))
		sic_low = np.zeros((len(signals),len(points)+2))
		sic_upp = np.zeros((len(signals),len(points)+2))
		roc_name = metric_NN_files_names[metric] if default_classifier=="NN" else "BDT"
		folder = folder_path(gen_direc, default_classifier, mode, default=True)
		for j,s in enumerate(signals):
			sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")
		plot_sic(ax[2], sic[:,index], sic_low[:,index], sic_upp[:,index], signals, "grey", "Default", linestyle="dashed")

	sic = np.zeros((len(classifiers), len(signals),10))
	sic_half = np.zeros((len(classifiers), len(signals),10))
	metric_values = np.zeros((len(classifiers), len(signals), 10))
	for i,c in enumerate(classifiers): 
		roc_name = metric_NN_files_names[metric] if c=="NN" else "BDT"
		is_NN = (c=="NN")
		folder = folder_path(gen_direc, c, mode, half=False, default=False, metric=metric)
		folder_half = folder_path(gen_direc, c, mode, half=True, default=False, metric=metric)
		name = mode+suffix

		for j,s in enumerate(signals):
			sic[i,j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/", return_full_array=True)
			sic_half[i,j] = read_ROC_SIC_1D(roc_name, points, folder_half + "Nsig_"+str(s)+"/", return_full_array=True)
			metric_values[i,j] = read_metric(folder_half+ "Nsig_"+str(s)+"/", metric, is_NN)
		plot_sic(ax[2], np.median(sic[i], axis=-1), np.percentile(sic[i], 16, axis=-1), np.percentile(sic[i], 84, axis=-1), signals, classifier_colors[c], c)
		plot_sic(ax[0], np.median(metric_values[i], axis=-1), np.percentile(metric_values[i], 16, axis=-1), np.percentile(metric_values[i], 84, axis=-1), signals, classifier_colors[c], c)
		plot_sic(ax[1], np.median(sic_half[i], axis=-1), np.percentile(sic_half[i], 16, axis=-1), np.percentile(sic_half[i], 84, axis=-1), signals, classifier_colors[c], c)
		
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

def plot_selection(ax2, mode, signals, signals_plot, gen_direc, loc="lower right", legend=False, plot_default=True, rotated=False, 
				   metrics="val_SIC", metric_opt="val_SIC", classifiers=["NN", "HGB", "AdaBoost"], default_hp=False):    
	'''
	Plot subplot of architecture selection
	'''

	suffix = "_rotated" if rotated else ""
	def folder_path(base, clf, mode, half=False, default=False, metric=None):
		hp_type = "default_hp" if default else "optimized_hp"
		half_suffix = "_half" if half else ""
		metric_path = f"{metric}/" if metric and not default else ""
		return f"{base}{clf}/{hp_type}/{mode}{suffix}{half_suffix}/{metric_path}"

	for metric in metrics:
		metric_opt = metric_opt or metric

		if plot_default:
			sic = np.zeros((len(signals),len(points)+2))
			sic_low = np.zeros((len(signals),len(points)+2))
			sic_upp = np.zeros((len(signals),len(points)+2))
			roc_name = metric_NN_files_names[metric_opt] if default_classifier=="NN" else "BDT"
			folder = folder_path(gen_direc, default_classifier, mode, default=True)
			for j,s in enumerate(signals):
				sic[j], sic_low[j], sic_upp[j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/")

		sic = np.zeros((len(classifiers), len(signals),10))
		sic_half = np.zeros((len(classifiers), len(signals),10))
		metric_values = np.zeros((len(classifiers), len(signals), 10))
		for i,c in enumerate(classifiers): 	
			roc_name = metric_NN_files_names[metric_opt] if c=="NN" else "BDT"
			is_NN = (c=="NN")
			folder = folder_path(gen_direc, c, mode, half=False, metric=metric_opt, default=default_hp)
			folder_half = folder_path(gen_direc, c, mode, half=True, metric=metric_opt, default=default_hp)

			for j,s in enumerate(signals):
				sic[i,j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/", return_full_array=True)
				sic_half[i,j] = read_ROC_SIC_1D(roc_name, points, folder_half + "Nsig_"+str(s)+"/", return_full_array=True)
				metric_values[i,j] = sic_half[i,j] if metric=="max_SIC" else read_metric(folder_half+ "Nsig_"+str(s)+"/", metric, is_NN)
			
		select = np.argmax(metric_maximize[metric]*metric_values, axis=0)
		sic_full_select = np.array([[sic[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
		med, p16, p84 = np.median(sic_full_select, axis=-1), np.percentile(sic_full_select, 16, axis=-1), np.percentile(sic_full_select, 84, axis=-1)
		plot_sic(ax2, med, p16, p84, signals, metric_colors[metric], metric_names[metric]+" selection")
	plot_end_1D_multiple(ax2, signals_plot, title = mode_name[mode], legend=legend, loc=loc, ylim=max_SIC_lims)


def plot_selection_multiple(signals, signals_plot, gen_direc, plotting_directory, modes = ["IAD", "cwola", "cathode"], rotated=False, 
								  loc="lower right", default_hp=False, metrics=["max_SIC", "val_loss", "val_SIC"], classifiers=["NN", "HGB", "AdaBoost"], 
								  mode_legend="IAD", metric_opt=None, plot_default=False, legend_param=True):
	'''
	Plot architecture selection for each mode
	'''
	fig, ax = plt.subplots(1,len(modes), figsize=(15,5))
	for i,mode in enumerate(modes): 
		legend = (mode==mode_legend and legend_param)
		plot_selection(ax[i], mode, signals, signals_plot, gen_direc, loc=loc, plot_default=plot_default, rotated=rotated, 
				metrics=metrics, metric_opt=metric_opt, classifiers=classifiers, legend=legend, default_hp=default_hp)
	fig.tight_layout()
	fig.savefig(plotting_directory+"1D_selection_all.pdf")

def plot_feature_selection_single_panel(mode, signals, signals_plot, gen_direc, plotting_directory, ylim=None, second_axis=True, loc="lower right", plot_default=True, rotated=False, metric="val_SIC", feature_sets=["baseline", "extended1", "extended2", "extended3", "DeltaR"], classifier="HGB"):    
	'''
	Plot feature selection
	'''
	fig, ax = plt.subplots(figsize=(7, 5))

	sic = np.zeros((len(feature_sets), len(signals),10))
	metric_values = np.zeros((len(feature_sets), len(signals), 10))
	for i,f in enumerate(feature_sets): 
		feature_set_file = "" if f=="baseline" else "_"+f
		roc_name = metric_NN_files_names[metric] if classifier=="NN" else "BDT"
		is_NN = (classifier=="NN")

		suffix = "_rotated" if rotated else ""
		folder = gen_direc+classifier+"/default_hp/"+mode+suffix+feature_set_file+"/"
		folder_half = gen_direc+classifier+"/default_hp/"+mode+suffix+"_half"+feature_set_file+"/"
		name = mode + suffix

		for j,s in enumerate(signals):
			sic[i,j] = read_ROC_SIC_1D(roc_name, points, folder + "Nsig_"+str(s)+"/", return_full_array=True)
			metric_values[i,j] = read_metric(folder_half+ "Nsig_"+str(s)+"/", metric, is_NN)
		plot_sic(ax, np.median(sic[i], axis=-1), np.percentile(sic[i], 16, axis=-1), np.percentile(sic[i], 84, axis=-1), signals, feature_set_colors[f], feature_set_names[f])

	select = np.argmax(metric_maximize[metric]*metric_values, axis=0)
	sic_full_select = np.array([[sic[select[i,j],i,j] for j in range(10)] for i in range(len(signals))])
		
	plot_sic(ax, np.median(sic_full_select, axis=-1), np.percentile(sic_full_select, 16, axis=-1), np.percentile(sic_full_select, 84, axis=-1), signals, "black", "Selected")
	plot_end_1D(fig, ax, signals_plot, name+"feature_selected_single_panel", ylim=40, loc=loc)

