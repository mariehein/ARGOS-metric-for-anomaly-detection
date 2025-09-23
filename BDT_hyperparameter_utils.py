import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import metric_utils as metrics

def classifier_training(X_train, Y_train, X_test, Y_test, args, run, X_eval=None, Y_eval=None, direc_run=None):
    np.random.seed(run)

    if args.classifier=="AdaBoost":
        hyperparameters = {"estimator": {"max_depth": np.random.randint(2, 15), 
                                        "max_leaf_nodes": np.random.randint(2, 100), 
                                        "min_samples_leaf": np.random.randint(1, 100), 
                                        "class_weight": 'balanced'},
        "learning_rate": 10**np.random.uniform(-4, -0.1), 
        "n_estimators": np.random.randint(2, 100)}
    else: 
        hyperparameters = {"learning_rate": 10**np.random.uniform(-4, -0.1),
                       "max_iter": np.random.randint(2,200),
                       "max_leaf_nodes": np.random.randint(2, 100),
                       "max_depth": np.random.randint(2, 15),
                       "l2_regularization": np.random.uniform(0,10),
                       "max_bins": np.random.randint(31, 255), 
                       "verbose": 0,
                       "validation_fraction": 0.5,
                       "class_weight": "balanced", 
                       "early_stopping": True,
                       }

    val_SIC=np.zeros((args.runs_per_hp_set,args.averaging_runs))
    max_SIC=np.zeros((args.runs_per_hp_set,args.averaging_runs))
    val_loss=np.zeros((args.runs_per_hp_set,args.averaging_runs))
    for j in range(args.runs_per_hp_set):
        for i in range(args.averaging_runs):
            np.random.seed(j*args.runs_per_hp_set+i)
            print("Tree number:", j*args.runs_per_hp_set+i)
            if args.classifier=="AdaBoost":
                # 50-50 split of data to get randomized ensemble
                X_train_local, X_val, Y_train_local, Y_val = train_test_split(X_train, Y_train, test_size=0.5)
                tree = AdaBoostClassifier(estimator=DecisionTreeClassifier(**hyperparameters["estimator"])
                                        , learning_rate=hyperparameters["learning_rate"]
                                        , n_estimators=hyperparameters["n_estimators"])
                _ = tree.fit(X_train_local, Y_train_local)
            else: 
                tree = HistGradientBoostingClassifier(**hyperparameters)
                _ = tree.fit(X_train, Y_train)
                _,X_val,_, Y_val = train_test_split(X_train, Y_train, test_size=0.5, stratify=Y_train, random_state=tree._random_seed)
            val_SIC[j,i], val_loss[j,i] = metrics.get_val_metrics(tree.predict_proba(X_val)[:,1], Y_val)
            max_SIC[j,i] = metrics.max_sic(tree.predict_proba(X_test)[:,1], Y_test)
    
    metrics.save_array_to_zip(args.directory+"hp_opt.zip", np.mean(val_loss, axis=1), "run"+str(args.hp_run_number)+"_val_loss")
    metrics.save_array_to_zip(args.directory+"hp_opt.zip", np.mean(val_SIC, axis=1), "run"+str(args.hp_run_number)+"_val_SIC")
    metrics.save_array_to_zip(args.directory+"hp_opt.zip", np.mean(max_SIC, axis=1), "run"+str(args.hp_run_number)+"_max_SIC")
    metrics.save_dict_to_zip(args.directory+"hp_opt.zip", hyperparameters, "run"+str(args.hp_run_number)+"_hp")

    
