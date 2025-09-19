import NN_model_utils as cl
import yaml
import os
import numpy as np
import io
from zipfile import ZipFile
import shutil

def save_array_to_zip(zip_path, array, name):
    """Save a numpy array into a zip as .npy"""
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    with ZipFile(zip_path, "a") as z:   # "a" = append
        z.writestr(f"results/{name}.npy", buf.read())

def save_dict_to_zip(zip_path, dictionary, name):
    """Save a Python dict into a zip as .yaml"""
    yaml_text = yaml.dump(dictionary)
    with ZipFile(zip_path, "a") as z:
        z.writestr(f"results/{name}.yaml", yaml_text)

def classifier_training(X_train, Y_train, X_test, Y_test, args, run, X_val=None, Y_val=None, direc_run=None):
    if direc_run is None:	
        direc_run=args.directory
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)
    np.random.seed(run)

    #Randomize train val set before run
    inds = np.arange(X_train.shape[0])
    np.random.shuffle(inds)
    X_train = X_train[inds]
    Y_train = Y_train[inds]

    params = {}
    params["lr"] = float(np.random.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]))
    params["batch_size"] = int(np.random.choice([64, 128, 256, 512, 1024, 2048, 5096]))
    params["layers"] = [64,64,64]
    params["epochs"] = int(30)
    params["dropout"] = float(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
    params["weight_decay"] = float(np.random.choice([0,1e-4, 1e-3, 1e-2, 1e-5]))
    params["momentum"] = float(np.random.choice([0.9, 0.99, 0.8, 0.95]))
    print(params)

    val_SIC=np.zeros((args.runs_per_hp_set,args.averaging_runs))
    max_SIC=np.zeros((args.runs_per_hp_set,args.averaging_runs))
    val_loss=np.zeros((args.runs_per_hp_set,args.averaging_runs))
    for j in range(args.runs_per_hp_set):
        for i in range(args.averaging_runs):
            model = cl.NeuralNetworkClassifier(save_path = direc_run, n_inputs=args.inputs, layers=params["layers"], early_stopping=False
                                        , val_split=0.5, lr = params["lr"], batch_size=params["batch_size"], epochs=params["epochs"]
                                        , dropout=params["dropout"], weight_decay=params["weight_decay"], momentum=params["momentum"], save_model=False)
            
            
            model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, X_val=X_val, y_val=Y_val)
            val_loss[j,i] = np.min(np.load(direc_run+"CLSF_val_losses.npy"))
            val_SIC[j,i] = np.max(np.load(direc_run+"CLSF_val_SIC.npy"))
            max_SIC[j,i] = np.max(np.load(direc_run+"CLSF_max_SIC.npy"))

    params["epochs"] = int(100)
    print(params)

    save_array_to_zip(args.directory+"hp_opt.zip", np.mean(val_loss, axis=1), "run"+str(args.hp_run_number)+"_val_loss.npy")
    save_array_to_zip(args.directory+"hp_opt.zip", np.mean(val_SIC, axis=1), "run"+str(args.hp_run_number)+"_val_SIC.npy")
    save_array_to_zip(args.directory+"hp_opt.zip", np.mean(max_SIC, axis=1), "run"+str(args.hp_run_number)+"_max_SIC.npy")
    save_dict_to_zip(args.directory+"hp_opt.zip", params, "run"+str(args.hp_run_number)+"_hp.yaml")
    shutil.rmtree(direc_run, ignore_errors=False, onerror=None)