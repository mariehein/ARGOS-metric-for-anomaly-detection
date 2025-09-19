import dataprep_utils as dp
import argparse 
import sys
import os

parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parameters to vary for the used runs
parser.add_argument('--mode', type=str, choices=["cathode", "cwola", "IAD", "supervised"], required=True)
parser.add_argument('--classifier', type=str, choices=["NN", "AdaBoost", "HGB"], required=True)
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--apply_random_rotation', default=False, action="store_true", help="Applies random rotation to input feature space")
parser.add_argument('--signal_number', type=int, default=1000, help="number of signal events")
parser.add_argument('--hp_run_number', required=True, type=int, help="start at 0, should be managed by bash file")

# Need to pass file locations
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--samples_file', type=str, default=None)

# Dataset arguments
parser.add_argument('--input_set', type=str, choices=["baseline", "baseline41" , "extended1", "extended2", "extended3", "extended7"])
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--include_DeltaR', default=False, action="store_true", help="appending Delta R to input feature set")

parser.add_argument('--signal_file', type=str, default=None, help="Specify different signal file")
parser.add_argument('--three_pronged', default=False, action="store_true", help="Activate three-pronged signal file")
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

# HP optimization arguments (Don't change for paper settings)
parser.add_argument('--runs_per_hp_set', default=10, type=int, help="Used to get error bands in plots")
parser.add_argument('--averaging_runs', default=None, type=int, help="Used to average runs to get more stable optimization")

args = parser.parse_args()

if args.classifier=="NN":
    import NN_hyperparameter_utils as cl
else: 
    import BDT_hyperparameter_utils as cl

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

if args.averaging_runs is None:
    if args.classifier=="NN":
        args.averaging_runs=1
    else:
        args.averaging_runs=10

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

if args.three_pronged:
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"
     
print(args)

X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
direc_run = args.directory+"run"+str(args.hp_run_number)+"/"
cl.classifier_training(X_train, Y_train, X_test, Y_test, args, args.hp_run_number, direc_run=direc_run)