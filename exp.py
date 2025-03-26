import numpy as np
import scipy as sc
from tqdm import tqdm
from scipy import linalg
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit as sigmoid
from scipy.special import log_expit
from joblib import Parallel, delayed

import pandas as pd

import bottleneck as bn

import datetime

import os

import numpy as np
from scipy import sparse

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



def load_xtx_binary(I):
    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))
    X_reg=train_data[:,:I]
    X = X_reg.toarray()
    #Xp = csr_matrix(np.where(X==-1,0,1))
    #Xn = csr_matrix(np.where(X==1,0,-1))
    #X = -1.0*csr_matrix.sign(X)
    #X = 2 * X - 1
    ####normalize users
    #nn=np.array(np.sum(X,axis=1)) [:,0]
    #nn=1.0/np.sqrt(nn)  # user weight normalized on diagonal, approx prop to nn  (off diag)
    #X=  sparse.spdiags(nn, 0, len(nn), len(nn)) * X
    ### remove mean  --> cov
    print (X_reg.shape)
    #XtX=np.array(X.T.dot(X).todense()) 

    return [X_reg.shape[0] , X_reg]


def load_xtx_sign(I):
    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))
    
    X=train_data[:,:I]
    X = X.toarray()
    #X = -1.0*csr_matrix.sign(X)
    X = 2 * X - 1
    ####normalize users
    #nn=np.array(np.sum(X,axis=1)) [:,0]
    #nn=1.0/np.sqrt(nn)  # user weight normalized on diagonal, approx prop to nn  (off diag)
    #X=  sparse.spdiags(nn, 0, len(nn), len(nn)) * X
    ### remove mean  --> cov
    print (X.shape)
    #XtX=np.array(X.T.dot(X).todense()) 

    return [X.shape[0] ,X,None]

#I = 20108
I = 10_000
U = 116677
userCnt , X= load_xtx_binary(I)
X_ = X.toarray()
Xt = (2*X_-1)


scale = 1


def ease(X, lam):
    print('multiplying matrix')
    G = X.T @ X
    diagIndices = np.diag_indices(G.shape[0])
    G = G + lam * np.eye(G.shape[0])
    print('inverting')
    P = linalg.inv(G)
    print('inverting complete')
    B = P / (-np.diag(P))
    B[diagIndices] = 0.0
    return B

identity = sc.sparse.eye(I)

def llog(w,item,lam):
    xw = Xt@w
    y = X_[:,item]
    return np.sum(np.logaddexp(0,(1-2*y)*xw)) + lam/2 * np.inner(w,w)

def dllog(w,item,lam):
    p = sigmoid(Xt@w) - X_[:,item]
    return p.T@Xt + lam*w


def hllog(w,item,lam):
    p = sigmoid(Xt@w)
    D = sc.sparse.diags(p*(1-p))
    return Xt.T@D@Xt.t + lam * identity




lam = 200.0
def exp(item):
    b = 0.0
    bounds = np.zeros((I,2))
    bounds[:,0] = -100
    bounds[:,1] = 100
    bounds[item,:] = np.array([b,b])
    # bounds = sc.optimize.Bounds(lb=bounds[:,0],ub=bounds[:,1])
    x0 = np.random.uniform(low=-0.0,high=0.0,size=I)
    x0[item] = b
    # clf = LogisticRegression(solver='L-BFGS-B', penalty = 'l2', C=lam)
    # clf.fit(Xt,X_[:,item],bounds=bounds)
    
    sol = minimize(
       fun = llog, x0=x0, args = (item,lam),method='l-bfgs-b',jac=dllog, bounds=bounds,
       options={'ftol':1e-12}
    )
    #print(sol, np.linalg.norm(sol.jac))
    return sol.x

print('running experiment')

W = Parallel(n_jobs=-3)(delayed(exp)(item) for item in tqdm(range(I)))
W = np.array(W)
np.save('weights_lbfgs_10000', W.T)


