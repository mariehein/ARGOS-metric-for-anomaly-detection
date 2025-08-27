import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import json
import os
import metric_utils as metrics
from sklearn.utils import class_weight


def classifier_training(X_train, Y_train, X_test, Y_test, args, run, direc_run=None):
    if direc_run is None:	
        direc_run=args.directory
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)

    class_weight_dict = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights = class_weight_dict[0]*(1-Y_train)+class_weight_dict[1]*Y_train

    print("\nTraining class weights: ", class_weight_dict)

    hyperparameters = json.load(open(args.hp_file))

    if args.use_half_statistics: 
        X_train, X_val, Y_train, Y_val, class_weights, val_weights = train_test_split(X_train, Y_train, class_weights, test_size=0.5)
        val_results = np.zeros((args.ensemble_over,len(X_val)))

    test_results = np.zeros((args.ensemble_over,len(X_test)))

    max_SIC = np.zeros((args.ensemble_over+1))
    val_SIC = np.zeros((args.ensemble_over+1))
    val_losses = np.zeros((args.ensemble_over+1))

    for j in range(args.ensemble_over):
        print("Tree number:", args.ensemble_over*run+j)
        np.random.seed(run*args.ensemble_over+j+1)
        if args.use_AdaBoost:
            X_train_local, X_val_local, Y_train_local, Y_val_local, class_weights_local, val_weights_local = train_test_split(X_train, Y_train, class_weights, test_size=0.5)
            tree = AdaBoostClassifier(**hyperparameters)
            results_f = tree.fit(X_train_local, Y_train_local)
            if not args.use_half_statistics:
                X_val, Y_val, val_weights = X_val_local, Y_val_local, val_weights_local
        else: 
            tree = HistGradientBoostingClassifier(**hyperparameters)
            print(tree.get_params())
            print(tree.max_depth)
            results_f = tree.fit(X_train, Y_train, sample_weight=class_weights)
            if not args.use_half_statistics:
                _, X_val, _, Y_val, _, val_weights = train_test_split(X_train, Y_train, class_weights, test_size=0.5, stratify=Y_train, random_state=tree._random_seed)
        if args.use_half_statistics:
            val_results[j] = tree.predict_proba(X_val)[:,1]
            val_losses[j] = metrics.log_loss(Y_val, val_results[j], sample_weight=val_weights)
            val_SIC[j] = metrics.val_roc(val_results[j], Y_val)
        else:
            val_results = tree.predict_proba(X_val)[:,1]
            val_losses[j] = metrics.log_loss(Y_val, val_results, sample_weight=val_weights)
            val_SIC[j] = metrics.val_roc(val_results, Y_val)
        test_results[j] = tree.predict_proba(X_test)[:,1]
        max_SIC[j] = metrics.calc_roc(test_results[j], Y_test)
        del tree
        del results_f
    test_results = np.mean(test_results, axis=0)
    max_SIC[-1] = metrics.calc_roc(test_results, Y_test)
    if args.use_half_statistics:
        val_results = np.mean(val_results, axis=0)
        val_SIC[-1] = metrics.val_roc(val_results, Y_val)
        val_losses[-1] = metrics.log_loss(Y_val, val_results, sample_weight=val_weights)
    print("AUC last epoch: %.3f" % metrics.plot_roc(test_results, Y_test,title=args.roc_name,directory=args.directory))

    np.save(direc_run+"max_SIC.npy", max_SIC)
    np.save(direc_run+"val_SIC.npy", val_SIC)
    np.save(direc_run+"val_losses.npy", val_losses)