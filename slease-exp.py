import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy import linalg
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit as sigmoid
from scipy.special import log_expit
from joblib import Parallel, delayed

import os

import numpy as np
from scipy import sparse

import pandas as pd

import bottleneck as bn

import datetime
from copy import deepcopy

import pickle

def save_pkl(obj, filename ):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL )
    
def load_pkl(filename ):
    with open(filename, 'rb') as f:
        return pickle.load(f)



DATA_DIR = '/efs/users/hsteck/public/data_for_ease/movielens20mio/'

pro_dir = os.path.join(DATA_DIR, 'pro_sg')


unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)

def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data


def load_xtx_binary():
    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))
    X_reg=train_data
    print (X_reg.shape)

    return [X_reg.shape[0] , X_reg]


I = 20108
U = 116677
print('loading data')
userCnt , X = load_xtx_binary()
print('converting data')
X_ = X.toarray()
print('transforming data')
Xt = (2*X_-1)


def llog(w,item,lam):
    xw = Xt@w
    y = X_[:,item]
    return np.sum(np.logaddexp(0,(1-2*y)*xw)) + lam/2 * np.inner(w,w)

def dllog(w,item,lam):
    p = sigmoid(Xt@w) - X_[:,item]
    return p.T@Xt + lam*w


def exp(item):
    lam = 200.0
    b = -2.0
    bounds = np.zeros((I,2))
    bounds[:,0] = -100
    bounds[:,1] = 100
    bounds[item,:] = np.array([b,b])
    x0 = np.random.uniform(low=-0.0,high=0.0,size=I)
    x0[item] = b
    sol = minimize(
       fun = llog, x0=x0, args = (item,lam),method='l-bfgs-b',jac=dllog, bounds=bounds,
       options={'gtol':1e-12,'ftol':1e-12}
    )
    return sol.x

print('running exp')
W = Parallel(n_jobs=80)(delayed(exp)(item) for item in tqdm(range(I)))
W = np.array(W)

np.save('weight.npy',W)


