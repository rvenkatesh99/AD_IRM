import torch
import numpy as np
import sys

if len(sys.argv) > 1:
    print("Using seed: ", sys.argv[1])
    seed_val = int(sys.argv[1])
else:
    print("Using default seed of 9")
    seed_val = 42


np.random.seed(seed_val)


def compute_metrics(model, feats, targs):

    # Predict on the test set
    test_preds = model.predict(feats)
    test_probs = model.predict_proba(feats)[:,1]

    test_targs = targs

    # Calculate the balanced accuracy
    from sklearn.metrics import balanced_accuracy_score
    balanced_accuracy = balanced_accuracy_score(test_targs, test_preds)
    print(f"Balanced Accuracy: {balanced_accuracy}")

    # Calculate the AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(test_targs, test_probs)
    print(f"AUC: {auc}")

    # Calculate the F1 score
    from sklearn.metrics import f1_score
    f1 = f1_score(test_targs, test_preds)
    print(f"F1 Score: {f1}")

    # Calculate the AUPRC
    from sklearn.metrics import average_precision_score
    auprc = average_precision_score(test_targs, test_probs)
    print(f"AUPRC: {auprc}")


data = torch.load('/project/ritchie/personal/rachitk/gnn-genomics-AD/data/adsp/no_apoe_covars/collated_tensors_withPheno_noSepAPOE.pt')

feats_all_samp, targs_all_samp = data['covariates'], data['target']

coalesced_genomic_data = data['data'].coalesce()
import scipy
scipy_sparse_genomic = scipy.sparse.csr_matrix((coalesced_genomic_data.values(), 
                                                (coalesced_genomic_data.indices()[0], coalesced_genomic_data.indices()[1])),
                                                shape=(coalesced_genomic_data.shape[0], coalesced_genomic_data.shape[1]))


scipy_sparse_all = scipy.sparse.hstack([scipy_sparse_genomic, scipy.sparse.csr_matrix(feats_all_samp)])

# Drop missing values for feats or targs
combined = np.concatenate((feats_all_samp, targs_all_samp.reshape(-1,1)), axis=1)
missing_feats = np.isnan(combined).any(axis=1)
combined = combined[~missing_feats]
feats, targs = combined[:,0:-1], combined[:,-1]

genomics = scipy_sparse_genomic[~missing_feats]
all_vals = scipy_sparse_all[~missing_feats]

# Get APOE only
# Check VarIDs for APOE4 and APOE2
# rs7412 and rs429358
rs7412_ind = data['VarID'].index('rs7412')
rs429358_ind = data['VarID'].index('rs429358')

apoe_only = genomics[:,[rs7412_ind, rs429358_ind]]


# Split the data into training and testing
rand_idx = np.random.permutation(feats.shape[0])
train_idx = rand_idx[0:int(0.8*feats.shape[0])]
test_idx = rand_idx[int(0.8*feats.shape[0]):]

train_agesex, test_agesex = feats[train_idx], feats[test_idx]
train_genomics, test_genomics = genomics[train_idx], genomics[test_idx]
train_all, test_all = all_vals[train_idx], all_vals[test_idx]
train_apoe, test_apoe = apoe_only[train_idx], apoe_only[test_idx]

train_targs, test_targs = targs[train_idx], targs[test_idx]

# Scale values based on training set (age-sex only)
# (but only age, not sex)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_agesex = scaler.fit_transform(train_agesex)
test_agesex = scaler.transform(test_agesex)




# Train a logistic regression model (age-sex only)
from sklearn.linear_model import LogisticRegressionCV
logreg = LogisticRegressionCV(random_state=9)
logreg = logreg.fit(train_agesex, train_targs)

# Predict on the test set
print("LogReg Age + Sex")
compute_metrics(logreg, test_agesex, test_targs)
print("")


# Train an elasticnet model on APOE by itself
from sklearn.linear_model import SGDClassifier
logreg_elastic_apoe = SGDClassifier(penalty = 'elasticnet', loss='log_loss',
                                        random_state=9)
logreg_elastic_apoe = logreg_elastic_apoe.fit(train_apoe, train_targs)

# Predict on the test set
print("LogReg Elastic APOE only")
compute_metrics(logreg_elastic_apoe, test_apoe, test_targs)
print("")


# Train an elasticnet model on APOE + Age/Sex
from sklearn.linear_model import SGDClassifier

train_apoe_agesex = np.concatenate([train_apoe.toarray(), train_agesex], axis=1)
test_apoe_agesex = np.concatenate((test_apoe.toarray(), test_agesex), axis=1)

logreg_elastic_apoe_all = SGDClassifier(penalty = 'elasticnet', loss='log_loss',
                                        random_state=9)
logreg_elastic_apoe_all = logreg_elastic_apoe_all.fit(train_apoe_agesex, train_targs)

# Predict on the test set
print("LogReg Elastic APOE + Age/Sex")
compute_metrics(logreg_elastic_apoe_all, test_apoe_agesex, test_targs)
print("")

# Train a neural network model (age-sex only)
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(hidden_layer_sizes=(256,128,64,32,16),
                   random_state=9)
nn = nn.fit(train_agesex, train_targs)

# Predict on the test set
print("NN Age + Sex")
compute_metrics(nn, test_agesex, test_targs)
print("")


# Train an elasticnet model on the genomic data only
from sklearn.linear_model import SGDClassifier
logreg_elastic_genomics = SGDClassifier(penalty = 'elasticnet', loss='log_loss',
                                        random_state=9)
logreg_elastic_genomics = logreg_elastic_genomics.fit(train_genomics, train_targs)

# Predict on the test set
print("LogReg Elastic Genomics")
compute_metrics(logreg_elastic_genomics, test_genomics, test_targs)
print("")


# Train an elasticnet model on the genomic data + age and sex
from sklearn.linear_model import SGDClassifier
logreg_elastic_all = SGDClassifier(penalty = 'elasticnet', loss='log_loss',
                                        random_state=9)
logreg_elastic_all = logreg_elastic_all.fit(train_all, train_targs)

# Predict on the test set
print("LogReg Elastic All (Genomics + Age + Sex)")
compute_metrics(logreg_elastic_all, test_all, test_targs)
print("")