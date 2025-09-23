import argparse
from zipfile import ZipFile
import numpy as np
import yaml
import os

parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parameters to vary for the used runs
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--file_save_directory', type=str, required=True)
parser.add_argument('--hp_runs_total', default=100, type=int, help="Change to desired number of HP sets tried")
parser.add_argument('--runs_per_hp_set', default=10, type=int, help="Change to number of runs per HP set")

args = parser.parse_args()

metric_names = ["val_loss", "val_SIC", "max_SIC"]
metric_best = [-1., 1., 1.]

if not os.path.exists(args.file_save_directory):
    os.makedirs(args.file_save_directory)

for j,m in enumerate(metric_names):
    metric = np.zeros((args.hp_runs_total, args.runs_per_hp_set))
    with ZipFile(args.directory+"hp_opt.zip", "r") as z:
        #print(z.infolist())
        for i in range(args.hp_runs_total):
            with z.open("run"+str(i)+"_"+m+".npy") as f:
                metric[i] = np.load(f)
        metric_best_run = np.argmax(metric_best[j]*metric, axis=0)
        for i in range(args.runs_per_hp_set):
            with z.open("run"+str(metric_best_run[i])+"_hp.yaml") as f:
                hp_file = yaml.safe_load(f)
            yaml.dump(hp_file, open(args.file_save_directory+"run"+str(i)+"_hp_"+m+".yaml", "w+"))

