import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
import yaml
import os
import metric_utils as metrics
from sklearn.tree import DecisionTreeClassifier


def classifier_training(X_train, Y_train, X_test, Y_test, args, run, X_eval=None, Y_eval=None, direc_run=None):
    if direc_run is None:	
        direc_run=args.directory
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)

    with open(args.cl_filename, 'r') as stream:
        hyperparameters = yaml.safe_load(stream)
        
    test_results = np.zeros((len(X_test)))
    if args.use_half_statistics:
        eval_results = np.zeros((len(X_eval)))

    for j in range(args.ensemble_over):
        print("Tree number:", args.ensemble_over*run+j)
        np.random.seed(run*args.ensemble_over+j+1)
        if args.use_AdaBoost:
            # 50-50 split of data to get randomized ensemble
            X_train_local, _, Y_train_local, _ = train_test_split(X_train, Y_train, test_size=0.5)
            tree = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(**hyperparameters["estimator"])
                                        , learning_rate=hyperparameters["learning_rate"]
                                        , n_estimators=hyperparameters["n_estimators"])
            results_f = tree.fit(X_train_local, Y_train_local)
        else: 
            tree = HistGradientBoostingClassifier(**hyperparameters)
            results_f = tree.fit(X_train, Y_train)

        if args.use_half_statistics:
            eval_results += tree.predict_proba(X_eval)[:,1]/args.ensemble_over
        test_results += tree.predict_proba(X_test)[:,1]/args.ensemble_over
        del tree
        del results_f
    
    # Evaluate on test set
    max_SIC = metrics.max_sic(test_results, Y_test)
    np.save(direc_run+"max_SIC.npy", max_SIC)
    print("AUC last epoch: %.3f" % metrics.plot_roc(test_results, Y_test,title=args.roc_name,directory=args.directory))

    if args.use_half_statistics:
        val_SIC, val_loss = metrics.get_val_metrics(eval_results, Y_eval)
        np.save(direc_run+"val_SIC.npy", val_SIC)
        np.save(direc_run+"val_loss.npy", val_loss)
