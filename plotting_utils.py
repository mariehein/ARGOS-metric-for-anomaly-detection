import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.cm as col
import warnings
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

classifier_colors = {"NN": "dodgerblue", "HGB": "red", "AdaBoost": "orange"}
metric_colors = {"val_sic": c_RWTH["r"], "val_loss": c_RWTH["tq"], "max_SIC": c_RWTH["p"]}
metric_names = {"val_sic": "ARGOS", "val_loss": "BCE", "max_SIC": "Max SIC"}
mode_name = {"IAD": "IAD", "cathode": "CATHODE", "cwola": "CWoLa"}
default_metric = "val_sic"

max_err=0.2

S = 20000
B = 312858

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
#plt.rcParams['text.usetex'] = True
plt.rcParams['figure.figsize'] = 3.5, 2.625
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

def read_ROC_SIC_1D(path, points, folder, N_runs=10, start_runs=0):
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
	return np.median(SIC_values,axis=0), np.percentile(SIC_values, 16, axis=0), np.percentile(SIC_values, 84, axis=0)

def plot_end_1D(fig, ax, sig, name, small=True, ylim=None, title=None, plotting_directory="plots/", loc="lower right"): 
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

    fig, ax = plt.subplots(figsize=(9,6))
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
    plot_end_1D(fig, ax, signals_plot, mode+"_default", title=title, plotting_directory=plotting_directory)