import numpy as np

def sigmoid(z):
    g = 1 / (1 + np.exp(-z)) #defining sigmoid function of logistic regression
    return g

def z_score(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    X_norm = (X - mu) / sigma #do Z score normalization on that set
    return X_norm

def sec_rate(predict_opt,pred_model):
    count = 0
    m = pred_model.shape[0]
    for i in range(m):
        if predict_opt[i] == pred_model[i]: #calculate the success rate of model
            count += 1
    rate = (count / m) * 100
    return rate
