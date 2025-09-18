import dataprep_utils as dp
import argparse
import os
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parameters to vary for the used runs
parser.add_argument('--mode', type=str, choices=["cathode", "cwola", "IAD", "supervised"], required=True)
parser.add_argument('--classifier', type=str, choices=["NN", "AdaBoost", "HGB"], required=True)
parser.add_argument('--cl_filename', type=str, default=None, help="For default HP change nothing")
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--use_half_statistics', default=False, action="store_true", help="Only uses half statistics for classifier, evaluates metrics on other half")
parser.add_argument('--apply_random_rotation', default=False, action="store_true", help="Applies random rotation to input feature space")

# Need to pass file locations
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--samples_file', type=str, default=None)

# Dataset arguments
parser.add_argument('--input_set', type=str, default="baseline", choices=["baseline", "baseline41" , "extended1", "extended2", "extended3", "extended7"])
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--include_DeltaR', default=False, action="store_true", help="appending Delta R to input feature set")

parser.add_argument('--signal_file', type=str, default=None, help="Specify different signal file")
parser.add_argument('--three_pronged', default=False, action="store_true", help="Activate three-pronged signal file")
parser.add_argument('--signal_number', type=int, default=1000, help="number of signal events")
parser.add_argument('--minmass', type=float, default=3.3, help="SR lower edge in TeV")
parser.add_argument('--maxmass', type=float, default=3.7, help="SR upper edge in TeV")
parser.add_argument('--cl_norm', default=True, action="store_false", help="Classifier input normalisation (mean=0 and std=1)")
parser.add_argument('--oversampling_factor', type=float, default=4, help="CATHODE oversampling factor")
parser.add_argument('--ssb_width', type=float, default=0.2, help="Short side band width for cwola hunting")
parser.add_argument('--gaussian_inputs', default=None, type=int, help="Adds N uninformative Gaussian features")
parser.add_argument('--N_normal_inputs', default=4, type=int, help="Needs to be set for Gaussian inputs")

# Seeds for dataset preparation
parser.add_argument('--set_seed', type=int, default=1, help="Changes seed used for shuffling")
parser.add_argument('--randomize_seed', default=False, action="store_true", help="Randomizes shuffling")
parser.add_argument('--randomize_signal', default=None, help="Set to int if signal randomization wanted")

#Classifier Arguments
parser.add_argument('--N_runs', type=int, default=10, help="Number of runs wanted for errors")
parser.add_argument('--start_at_run', type=int, default=0, help="Allows restart at higher run numbers")
parser.add_argument('--N_best_epochs', type=int, default=10, help="NN best epoch averaging")

args = parser.parse_args()

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

if args.cl_filename==None:
    args.cl_filename = "hp_"+args.classifier+"_default.yaml"
    
if args.classifier=="NN":
    import NN_utils as cl
else: 
    import BDT_utils as cl

if args.input_set=="extended1":
    args.inputs=10
elif args.input_set=="extended1_small":
    args.inputs=6
elif args.input_set=="extended2":
    args.inputs=12
elif args.input_set=="extended3":
    args.inputs=56
elif args.input_set=="extended7":
    args.inputs=20

if args.include_DeltaR:
    args.inputs+=1

if args.gaussian_inputs is not None:
    args.inputs+=args.gaussian_inputs

if args.three_pronged: # change if not run on HPC RWTH
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"
     
print(args)

if not args.randomize_seed:
    X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
    if args.use_half_statistics:
        X_train, X_eval, Y_train, Y_eval = train_test_split(X_train, Y_train, test_size=0.5, shuffle=True)
    else: X_eval, Y_eval = None, None
for i in range(args.start_at_run, args.N_runs):
    print()
    print("------------------------------------------------------")
    print()
    print("Classifier run no. ", i)
    print()
    direc_run = args.directory+"run"+str(i)+"/"
    if args.randomize_seed or args.randomize_signal is not None:
        args.set_seed = i
        if args.randomize_signal is not None:
            args.randomize_signal = i
        X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
        if args.use_half_statistics:
            X_train, X_eval, Y_train, Y_eval = train_test_split(X_train, Y_train, test_size=0.5, shuffle=True)
        else: X_eval, Y_eval = None, None
    cl.classifier_training(X_train, Y_train, X_test, Y_test, args, i, X_eval=X_eval, Y_eval=Y_eval, direc_run=direc_run)