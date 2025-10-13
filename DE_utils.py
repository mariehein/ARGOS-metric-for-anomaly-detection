import numpy as np
import os
import yaml
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from DE_model_utils import ConditionalNormalizingFlow
from DE_model_utils import LogitScaler

def sample(model, m, n_samples, scaler):
    m_samples = np.random.choice(m, n_samples, replace=True).reshape((n_samples, 1))
    print(m_samples.shape, n_samples)
    X_samples = model.sample(m=m_samples)[:,0,:]
    print(X_samples.shape)
    X_samples = scaler.inverse_transform(X_samples)
    samples = np.hstack([m_samples, X_samples])
    return samples

def run_DE(args, innerdata, outerdata, direc_run):
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)

    with open(args.DE_filename, 'r') as stream:
        hyperparameters = yaml.safe_load(stream)

    train_data, val_data = train_test_split(outerdata, test_size=0.5, shuffle=True, random_state=args.set_seed)
    scaler = make_pipeline(LogitScaler(), StandardScaler())
    m_train = train_data[:,0:1]
    X_train = scaler.fit_transform(train_data[:,1:])
    m_val = val_data[:,0:1]
    X_val = scaler.transform(val_data[:,1:])
    
    flow_model = ConditionalNormalizingFlow(save_path=direc_run,
                                            num_inputs=X_train.shape[1],
                                            early_stopping=False,
                                            verbose=False, num_blocks=int(hyperparameters["blocks"]),
                                            batch_norm=bool(hyperparameters["use_batch_norm"]), 
                                            lr=float(hyperparameters["learning_rate"]), 
                                            batch_size=int(hyperparameters["batch_size"]), num_layers=int(hyperparameters["layers"]),
                                            epochs=int(hyperparameters["epochs"]), num_hidden=int(hyperparameters["hidden"])
                                            )

    flow_model.fit(X_train, m_train, X_val, m_val)
    np.save(direc_run+"samples_inner.npy",sample(flow_model, innerdata[:,0], n_samples=args.N_samples, scaler=scaler))
    np.save(direc_run+"samples_outer.npy",sample(flow_model, outerdata[:,0], n_samples=args.N_samples, scaler=scaler))
