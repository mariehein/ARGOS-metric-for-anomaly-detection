import NN_utils as cl
import yaml
import os
import numpy as np

def average_preds_metric(preds, metric, N_epochs=10):
    best_epochs = np.argsort(metric)[:N_epochs]
    return np.mean(preds[best_epochs], axis=0)

def classifier_training(X_train, Y_train, X_test, Y_test, args, run, X_val=None, Y_val=None, direc_run=None):
    if direc_run is None:	
        direc_run=args.directory
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)
    np.random.seed(run)

    params = {}
    params["lr"] = float(np.random.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]))
    params["batch_size"] = int(np.random.choice([64, 128, 256, 512, 1024, 2048, 5096]))
    #num_layers = np.random.randint(2,7)
    #layers = []
    #node_options = [16, 32, 64, 128]
    #for i in range(num_layers):
    #    layers.append(int(np.random.choice(node_options)))
    params["layers"] = [64,64,64]#layers
    params["epochs"] = int(30)
    params["dropout"] = float(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
    params["weight_decay"] = float(np.random.choice([0,1e-4, 1e-3, 1e-2, 1e-5]))
    params["momentum"] = float(np.random.choice([0.9, 0.99, 0.8, 0.95]))
    print(params)

    model = cl.NeuralNetworkClassifier(save_path = direc_run, n_inputs=args.inputs, layers=params["layers"], early_stopping=False
                                       , val_split=0.5, lr = params["lr"], batch_size=params["batch_size"], epochs=params["epochs"]
                                       , dropout=params["dropout"], weight_decay=params["weight_decay"], momentum=params["momentum"], save_model=False)

    params["epochs"] = int(100)
    print(params)
    with open(direc_run+"hp.yaml", 'w+') as stream:
        yaml.dump(params, stream)

    model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, X_val=X_val, y_val=Y_val)

def classifier_training_averaging(X_train, Y_train, X_test, Y_test, args, run, X_val=None, Y_val=None, direc_run=None):
    if direc_run is None:	
        direc_run=args.directory
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)
    np.random.seed(run)

        #num_layers = np.random.randint(2,7)
    #layers = []
    #node_options = [16, 32, 64, 128]
    #for i in range(num_layers):
    #    layers.append(int(np.random.choice(node_options)))

    params = {}
    params["lr"] = float(np.random.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]))
    params["batch_size"] = int(np.random.choice([64, 128, 256, 512, 1024, 2048, 5096]))
    params["layers"] = [64,64,64]#layers
    params["epochs"] = int(30)
    params["dropout"] = float(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]))
    params["weight_decay"] = float(np.random.choice([0,1e-4, 1e-3, 1e-2, 1e-5]))
    params["momentum"] = float(np.random.choice([0.9, 0.99, 0.8, 0.95]))
    print(params)

    averaging_runs = 10
    val_SIC=np.zeros(averaging_runs)
    max_SIC=np.zeros(averaging_runs)
    val_loss=np.zeros(averaging_runs)
    for i in range(averaging_runs):
        model = cl.NeuralNetworkClassifier(save_path = direc_run, n_inputs=args.inputs, layers=params["layers"], early_stopping=False
                                       , val_split=0.5, lr = params["lr"], batch_size=params["batch_size"], epochs=params["epochs"]
                                       , dropout=params["dropout"], weight_decay=params["weight_decay"], momentum=params["momentum"], save_model=False)
        
        
        model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, X_val=X_val, y_val=Y_val)
        val_loss[i] = np.min(np.load(direc_run+"CLSF_val_losses.npy"))
        val_SIC[i] = np.max(np.load(direc_run+"CLSF_val_SIC.npy"))
        max_SIC[i] = np.max(np.load(direc_run+"CLSF_max_SIC.npy"))
        
    np.save(direc_run+"average_val_loss.npy", np.mean(val_loss))
    np.save(direc_run+"average_val_SIC.npy", np.mean(val_SIC))
    np.save(direc_run+"average_max_SIC.npy", np.mean(max_SIC))

    params["epochs"] = int(100)
    print(params)
    with open(direc_run+"hp.yaml", 'w+') as stream:
        yaml.dump(params, stream)
