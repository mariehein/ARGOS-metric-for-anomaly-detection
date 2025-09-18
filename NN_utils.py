import NN_model_utils as NN
import metric_utils as metrics
import yaml
import os
import numpy as np
import json

def average_preds_metric(preds, metric, N_epochs=10):
    best_epochs = np.argsort(metric)[:N_epochs]
    return np.mean(preds[best_epochs], axis=0)

def classifier_training(X_train, Y_train, X_test, Y_test, args, run, X_eval=None, Y_eval=None, direc_run=None):
    if direc_run is None:	
        direc_run=args.directory
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)

    with open(args.cl_filename, 'r') as stream:
        params = yaml.safe_load(stream)
    print(params)

    model = NN.NeuralNetworkClassifier(save_path = direc_run, n_inputs=args.inputs, layers=params["layers"], early_stopping=False
                                       , val_split=0.5, lr = params["lr"], batch_size=params["batch_size"], epochs=params["epochs"]
                                       , dropout=params["dropout"], weight_decay=params["weight_decay"], momentum=params["momentum"])
    
    model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test)

    test_preds = model.get_all_predictions(X_test)
    print("AUC val SIC: %.3f" % metrics.plot_roc(average_preds_metric(test_preds, -np.load(model._val_SIC_path())), Y_test, title="val_sic",directory=args.directory))
    print("AUC max SIC: %.3f" % metrics.plot_roc(average_preds_metric(test_preds, -np.load(model._max_SIC_path())), Y_test, title="max_sic",directory=args.directory))
    print("AUC val loss: %.3f" % metrics.plot_roc(average_preds_metric(test_preds, np.load(model._val_loss_path())), Y_test, title="val_loss",directory=args.directory))
    del test_preds

    if X_eval is not None: 
        val_preds = model.get_all_predictions(X_eval)
        val_SIC = {}
        val_loss = {}
        val_SIC["val_SIC"], val_loss["val_SIC"] = metrics.get_val_metrics(average_preds_metric(val_preds, -np.load(model._val_SIC_path())), Y_eval)
        val_SIC["max_SIC"], val_loss["max_SIC"] = metrics.get_val_metrics(average_preds_metric(val_preds, -np.load(model._max_SIC_path())), Y_eval)
        val_SIC["val_loss"], val_loss["val_loss"] = metrics.get_val_metrics(average_preds_metric(val_preds, np.load(model._val_loss_path())), Y_eval)
        json.dump(val_SIC, open(direc_run+"val_SIC_averaged.json", 'w'))
        json.dump(val_loss, open(direc_run+"val_loss_averaged.json", 'w'))