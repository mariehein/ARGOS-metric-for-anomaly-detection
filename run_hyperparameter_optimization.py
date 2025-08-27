import NN_hyperparameter_utils as tr
import dataprep_utils as dp
import argparse 
import sys
import os

parser = argparse.ArgumentParser(
    description='Run the full CATHODE analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', type=str, choices=["cathode", "cwola", "IAD", "supervised"], required=True)
parser.add_argument('--input_set', type=str, choices=["baseline", "baseline41" , "extended1", "extended2", "extended3", "extended7"])
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--include_DeltaR', default=False, action="store_true", help="appending Delta R to input feature set")
parser.add_argument('--cl_filename', type=str, default="classifier4.yml")

#Data Preparation Arguments
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--samples_file', type=str, default=None)
parser.add_argument('--signal_file', type=str, default=None, help="Specify different signal file")
parser.add_argument('--three_pronged', default=False, action="store_true", help="Activate three-pronged signal file")
parser.add_argument('--signal_percentage', type=float, default=None)
parser.add_argument('--signal_number', type=int, default=None)
parser.add_argument('--minmass', type=float, default=3.3)
parser.add_argument('--maxmass', type=float, default=3.7)
parser.add_argument('--oversampling_factor', type=float, default=4)
parser.add_argument('--ssb_width', type=float, default=0.2)
parser.add_argument('--gaussian_distortion', type=float, default=None)
parser.add_argument('--cl_norm', default=True, action="store_false")
parser.add_argument('--gaussian_inputs', default=None, type=int)
parser.add_argument('--supervised_normal_signal', default=False, action="store_true")
parser.add_argument('--N_normal_inputs', default=4, type=int)
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--randomize_seed', default=False, action="store_true")
parser.add_argument('--randomize_signal', default=None)
parser.add_argument('--average_metrics', default=False, action="store_true")

#Classifier Arguments
parser.add_argument('--use_class_weights', default=True, action="store_false")
parser.add_argument('--use_val_weights', default=True, action="store_false")
parser.add_argument('--N_runs', type=int, default=10)
parser.add_argument('--start_at_run', type=int, default=0)
parser.add_argument('--N_best_epochs', type=int, default=10)

args = parser.parse_args()

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

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

if not args.randomize_seed:
    X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
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
    if args.average_metrics:
        tr.classifier_training_averaging(X_train, Y_train, X_test, Y_test, args, i, direc_run=direc_run)
    else:
        tr.classifier_training(X_train, Y_train, X_test, Y_test, args, i, direc_run=direc_run)