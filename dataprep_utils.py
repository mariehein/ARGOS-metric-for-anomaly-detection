import numpy as np
from scipy.special import logit, expit
import pandas as pd
import warnings
from scipy.stats import special_ortho_group

def shuffle_XY(X,Y):
    seed_int=np.random.randint(300)
    np.random.seed(seed_int)
    np.random.shuffle(X)
    np.random.seed(seed_int)
    np.random.shuffle(Y)
    return X,Y

class no_logit_norm:
	def __init__(self,array):
		self.mean = np.mean(array, axis=0)
		self.std = np.std(array, axis=0)

	def forward(self,array0):
		return (np.copy(array0)-self.mean)/self.std, np.ones(len(array0),dtype=bool)

	def inverse(self,array0):
		return np.copy(array0)*self.std+self.mean

def make_features_baseline(features, label_arr, m2=False):
    E_part = np.sqrt(features[:,0]**2+features[:,1]**2+features[:,2]**2+features[:,3]**2)+np.sqrt(features[:,7]**2+features[:,8]**2+features[:,9]**2+features[:,10]**2)
    p_part2 = (features[:,0]+features[:,7])**2+(features[:,1]+features[:,8])**2+(features[:,2]+features[:,9])**2
    m_jj = np.sqrt(E_part**2-p_part2)
    ind=np.array(features[:,10]> features[:,3]).astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        if m2:
            feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, features[:,10]*1e-3, features[:, 5]/features[:,4], features[:, 12]/features[:,11], features[:, 6]/features[:,5], features[:, 13]/features[:,12], label_arr])
            feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, features[:,3]*1e-3, features[:, 12]/features[:,11], features[:, 5]/features[:,4], features[:, 13]/features[:,12], features[:, 6]/features[:,5], label_arr])
        else:
            feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, (features[:,10]-features[:, 3])*1e-3, features[:, 5]/features[:,4], features[:, 12]/features[:,11], features[:, 6]/features[:,5], features[:, 13]/features[:,12], label_arr])
            feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, (features[:,3]-features[:, 10])*1e-3, features[:, 12]/features[:,11], features[:, 5]/features[:,4], features[:, 13]/features[:,12], features[:, 6]/features[:,5], label_arr])
    return np.nan_to_num(feat1*ind+feat2*(np.ones(len(ind))-ind)).T

def make_features_extended12(features_j1, features_j2, label_arr, set, m2=False):
    E_part2 = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2+features_j1[:,2]**2+features_j1[:,3]**2)+np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2+features_j2[:,2]**2+features_j2[:,3]**2)
    p_part2 = (features_j1[:,0]+features_j2[:,0])**2 + (features_j1[:,1]+features_j2[:,1])**2 + (features_j1[:,2]+features_j2[:,2])**2
    m_jj = np.sqrt(E_part2**2-p_part2)

    ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
    f_j1 = np.copy(features_j1)
    features_j1[ind_not] = features_j2[ind_not]
    features_j2[ind_not] = f_j1[ind_not]
    del f_j1	

    if set=="extended2":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            features = np.zeros((len(m_jj), 14))
            features[:,0] = m_jj * 1e-3
            features[:,1] = features_j1[:, 3] * 1e-3
            if m2:
                features[:,2] = features_j2[:, 3] *1e-3
            else:
                features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
            for i in range(5):
                features[:,3+2*i] = features_j1[:,4+i]
                features[:,3+2*i+1] = features_j2[:,4+i]
            features[:,-1] = label_arr
    elif set=="extended1":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
            warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
            inputs = 12
            features = np.zeros((len(m_jj), inputs))
            features[:,0] = m_jj * 1e-3
            features[:,1] = features_j1[:, 3] * 1e-3
            if m2:
                features[:,2] = features_j2[:, 3] *1e-3
            else:
                features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
            for i in range(4):
                features[:,3+2*i] = features_j1[:,5+i]/features_j1[:,4+i]
                features[:,3+2*i+1] = features_j2[:,5+i]/features_j2[:,4+i]
            features[:,-1] = label_arr

    return np.nan_to_num(features)

def make_features_extended3(pandas_file, label_arr, set, m2=False):
    features_j1 = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1']], dtype=np.float32)
    features_j2 = np.array(pandas_file[['pxj2', 'pyj2', 'pzj2', 'mj2']], dtype=np.float32)

    E_part2 = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2+features_j1[:,2]**2+features_j1[:,3]**2)+np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2+features_j2[:,2]**2+features_j2[:,3]**2)
    p_part2 = (features_j1[:,0]+features_j2[:,0])**2 + (features_j1[:,1]+features_j2[:,1])**2 + (features_j1[:,2]+features_j2[:,2])**2
    m_jj = np.sqrt(E_part2**2-p_part2)

    beta = [5, 1, 2]
    jet = ["j1", "j2"]
    to_subjettiness=9
    subjettinesses = np.zeros((len(m_jj),2,to_subjettiness*3))
    for k,b in enumerate(beta):
        for l, j in enumerate(jet):
            names = ["tau"+str(i)+j+"_"+str(b) for i in range(1,to_subjettiness+1)]
            subjettinesses[:,l,k*to_subjettiness:(k+1)*to_subjettiness]=np.array(pandas_file[names], dtype=np.float32)

    ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
    f_j1 = np.copy(features_j1)
    features_j1[ind_not] = features_j2[ind_not]
    features_j2[ind_not] = f_j1[ind_not]
    del f_j1
    s = np.copy(subjettinesses)
    subjettinesses[ind_not,0] = subjettinesses[ind_not,1] 
    subjettinesses[ind_not,1] = s[ind_not,0] 
    del s

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        features = np.zeros((len(m_jj), 4+to_subjettiness*6))
        features[:,0] = m_jj * 1e-3
        features[:,1] = features_j1[:, 3] * 1e-3
        if m2:
            features[:,2] = features_j2[:, 3] *1e-3
        else:
            features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
        features[:,3:3+subjettinesses.shape[-1]] = subjettinesses[:,0]
        features[:,3+subjettinesses.shape[-1]:3+subjettinesses.shape[-1]*2]= subjettinesses[:,1]
        features[:,-1] = label_arr

    return np.nan_to_num(features)

def make_features_extended7(pandas_file, label_arr, m2=False):
    features_j1 = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1']], dtype=np.float32)
    features_j2 = np.array(pandas_file[['pxj2', 'pyj2', 'pzj2', 'mj2']], dtype=np.float32)

    E_part2 = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2+features_j1[:,2]**2+features_j1[:,3]**2)+np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2+features_j2[:,2]**2+features_j2[:,3]**2)
    p_part2 = (features_j1[:,0]+features_j2[:,0])**2 + (features_j1[:,1]+features_j2[:,1])**2 + (features_j1[:,2]+features_j2[:,2])**2
    m_jj = np.sqrt(E_part2**2-p_part2)

    beta = [7]
    jet = ["j1", "j2"]
    to_subjettiness=9
    subjettinesses = np.zeros((len(m_jj),2,to_subjettiness*3))
    for k,b in enumerate(beta):
        for l, j in enumerate(jet):
            names = ["tau"+str(i)+j+"_"+str(b) for i in range(1,to_subjettiness+1)]
            subjettinesses[:,l,k*to_subjettiness:(k+1)*to_subjettiness]=np.array(pandas_file[names], dtype=np.float32)

    ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
    f_j1 = np.copy(features_j1)
    features_j1[ind_not] = features_j2[ind_not]
    features_j2[ind_not] = f_j1[ind_not]
    del f_j1
    s = np.copy(subjettinesses)
    subjettinesses[ind_not,0] = subjettinesses[ind_not,1] 
    subjettinesses[ind_not,1] = s[ind_not,0] 
    del s
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        features = np.zeros((len(m_jj), 4+to_subjettiness*6))
        features[:,0] = m_jj * 1e-3
        features[:,1] = features_j1[:, 3] * 1e-3
        if m2:
            features[:,2] = features_j2[:, 3] *1e-3
        else:
            features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
        features[:,3:3+subjettinesses.shape[-1]] = subjettinesses[:,0]
        features[:,3+subjettinesses.shape[-1]:3+subjettinesses.shape[-1]*2]= subjettinesses[:,1]
        features[:,-1] = label_arr

    return np.nan_to_num(features)

def file_loading(filename, args, labels=True, signal=0):
    pandas_file = pd.read_hdf(filename)
    if labels:
        label_arr = np.array(pandas_file['label'], dtype=float)
    else: 
        label_arr = np.ones((len(pandas_file['pxj1'])), dtype=float)*signal

    if args.input_set == "baseline":
        features = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']], dtype=float)
        features = make_features_baseline(features, label_arr)
    elif args.input_set == "baseline41":
        features = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau4j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau4j2_1', 'tau3j2_1']], dtype=float)
        features = make_features_baseline(features, label_arr)
    elif args.input_set in ["extended1", "extended2"]:
        features_j1 = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'tau4j1_1', 'tau5j1_1']], dtype=float)
        features_j2 = np.array(pandas_file[['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1', 'tau4j2_1', 'tau5j2_1']], dtype=float)
        features = make_features_extended12(features_j1, features_j2, label_arr, args.input_set)
        del features_j1, features_j2
    elif args.input_set in ["extended3"]:
        features = make_features_extended3(pandas_file, label_arr, args.input_set)
    elif args.input_set=="extended7":
        features = make_features_extended7(pandas_file, label_arr)
    del pandas_file
    return features

def DR(filename, labels=True):
    """
    Calculate DeltaR if args.include_DeltaR is true
    """
    if labels:
        features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']],dtype=float)
    else: 
        features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']],dtype=float)
        features = np.concatenate((features,np.zeros((len(features),1))),axis=1)
    Dphi = np.arccos((features[:,0]*features[:,7]+features[:,1]*features[:,8])/(np.sqrt(features[:,1]**2+features[:,0]**2)*np.sqrt(features[:,7]**2+features[:,8]**2)))
    eta1 = np.arcsinh(features[:,2]/np.sqrt(features[:,1]**2+ features[:,0]**2))
    eta2 = np.arcsinh(features[:,9]/np.sqrt(features[:,7]**2+ features[:,8]**2))
    DR = np.sqrt((Dphi)**2 + (eta1-eta2)**2)
    return DR

def classifier_data_prep(args, samples=None):
    data = file_loading(args.data_file, args)
    extra_bkg = file_loading(args.extrabkg_file, args, labels=False)
    if args.signal_file is not None: 
        data_signal = file_loading(args.signal_file, args, labels=False, signal=1)

    if args.signal_file is not None: 
        sig = data_signal
    else:
        sig = data[data[:,-1]==1]

    if args.include_DeltaR:
        data_DR = DR(args.data_file)
        data = np.concatenate((data[:,:args.inputs],np.array([data_DR]).T, data[:,args.inputs:]),axis=1)
        extra_bkg_DR = DR(args.extrabkg_file)
        extra_bkg = np.concatenate((extra_bkg[:,:args.inputs],np.array([extra_bkg_DR]).T, extra_bkg[:,args.inputs:]),axis=1)
        if args.signal_file is not None:
            sig_DR = DR(args.signal_file, labels=False)
            sig = np.concatenate((sig[:,:args.inputs],np.array([sig_DR]).T, sig[:,args.inputs:]),axis=1)
        else:
            sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]
    print(len(bkg), len(sig))
    print(args.gaussian_inputs)

    n_sig=args.signal_number

    if args.randomize_signal is not None:
        np.random.seed(args.randomize_signal)
        np.random.shuffle(sig)

    data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
    np.random.seed(args.set_seed)
    np.random.shuffle(data_all)
    extra_sig = sig[n_sig:]
    innersig_mask = (extra_sig[:,0]>args.minmass) & (extra_sig[:,0]<args.maxmass)
    inner_extra_sig = extra_sig[innersig_mask]

    innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
    innerdata = data_all[innermask]
    outerdata = data_all[~innermask]

    if args.density_estimation: 
        np.save(args.directory+"innerdata.npy", innerdata)
        np.save(args.directory+"outerdata.npy", outerdata)
        return innerdata[:, :args.inputs+1], outerdata[:, :args.inputs+1]

    if args.mode=="cwola":
        mask = (outerdata[:,0]>args.minmass-args.ssb_width) & (outerdata[:,0]<args.maxmass+args.ssb_width)
        samples_train = outerdata[mask]
    elif args.mode=="cathode":
        if args.samples_file is None: 
            raise ValueError("Samples file can not be None for cathode")
        samples_train = np.load(args.samples_file)[:int(len(innerdata)*args.oversampling_factor)]
        samples_add = np.load(args.samples_file)[int(len(innerdata)*args.oversampling_factor):int(len(innerdata)*args.oversampling_factor)+400000]
        samples_train = np.concatenate((samples_train, np.zeros((len(samples_train),1))), axis=1)

    extrabkg1 = extra_bkg[:312858]
    extrabkg2 = extra_bkg[312858:]

    if args.mode=="IAD":
        samples_train = extrabkg1[40000:]

    if args.mode=="supervised":
        sig_train = innerdata[:120000]
        sig_train = sig_train[sig_train[:,-1]==1]
        X_train = np.concatenate((samples_train, sig_train), axis=0)
        Y_train = X_train[:,-1]
        if args.gaussian_inputs is not None:
            gauss = np.random.normal(size=(len(X_train),args.inputs-args.N_normal_inputs))
            X_train = np.concatenate((X_train[:,1:args.N_normal_inputs+1],gauss), axis=1)
        else:
            X_train = X_train[:,1:args.inputs+1]
    else:
        if args.gaussian_inputs is not None:
            X_train = np.concatenate((innerdata[:120000,1:args.N_normal_inputs+1],samples_train[:,1:args.N_normal_inputs+1]),axis=0)
            gauss = np.random.normal(size=(len(X_train),args.inputs-args.N_normal_inputs))
            X_train = np.concatenate((X_train,gauss), axis=1)
            print(X_train.shape)
        else:
            X_train = np.concatenate((innerdata[:120000,1:args.inputs+1],samples_train[:,1:args.inputs+1]),axis=0)
        Y_train = np.concatenate((np.ones(len(X_train)-len(samples_train)),np.zeros(len(samples_train))),axis=0)		

    X_train, Y_train = shuffle_XY(X_train, Y_train)

    X_test = np.concatenate((extrabkg2,inner_extra_sig[:20000],extrabkg1[:40000]))
    Y_test = X_test[:,-1]
    if args.gaussian_inputs is not None:
        gauss = np.random.normal(size=(len(X_test),args.inputs-args.N_normal_inputs))
        X_test = np.concatenate((X_test[:,1:args.N_normal_inputs+1],gauss), axis=1)
    else:
        X_test = X_test[:,1:args.inputs+1]

    if args.apply_random_rotation: 
        rotation_matrix = special_ortho_group.rvs(dim=args.inputs, random_state=args.set_seed)
        X_train = X_train @ rotation_matrix
        X_test = X_test @ rotation_matrix
    print(X_train.shape, X_test.shape)

    if args.cl_norm:
        normalisation = no_logit_norm(X_train)
        X_train, _ = normalisation.forward(X_train)
        X_test, _ = normalisation.forward(X_test)

    print("Train set: ", len(X_train), "; Test set: ", len(X_test))

    if args.unhelpful_features:
        X_train = X_train[:,-4:]
        X_test = X_test[:,-4:]

    if args.mode == "cathode":
        if args.cl_norm:
            samples_add, _ = normalisation.forward(samples_add[:,1:args.inputs+1])
        X_eval_add = np.concatenate((samples_add, X_test[Y_test==0]), axis=0)
        Y_eval_add =  np.append(np.zeros(len(samples_add)), np.ones(len(X_test[Y_test==0])))
    else: 
        X_eval_add = None
        Y_eval_add = None

    return X_train, Y_train, X_test, Y_test, X_eval_add, Y_eval_add
