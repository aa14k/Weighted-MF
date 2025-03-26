import utils as ut
import scipy as sc
import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import cg
from scipy.special import expit as sigmoid
from joblib import Parallel, delayed



#path = '/efs/users/hsteck/public/datasets/netflix_prize_data/pro_sg'
#path = '/efs/users/hsteck/public/datasets/msd_data/pro_sg'
path = '/efs/users/hsteck/public/datasets/movielens20mio/pro_sg'

print(path)
X = ut.load_train_data(path)
X_ = X.toarray()
users,items = X.shape[0],X.shape[1]

def irls_optimized(item, lam=1.0, max_iter=50, tol=1e-12):
    items = X.shape[1]
    mask = np.ones(items, dtype=bool)
    mask[item] = False
    # Precompute these outside the loop
    diag_indices = np.diag_indices(items-1)
    w = np.zeros(items-1)
    for _ in (range(max_iter)):
        xw = X[:, mask].dot(w)
        p = sigmoid(xw)  # sigmoid function
        S = p * (1 - p)
        
        # Compute hessian efficiently
        hess = (X[:, mask].T.multiply(S).dot(X[:, mask])).toarray()
        hess[diag_indices] += lam
        
        # Compute gradient efficiently
        grad = X[:, mask].T.dot(p - X_[:,item]) + lam * w
        
        
        step = np.linalg.solve(hess, grad)
            
        beta = max(1e-32, np.linalg.norm(step))
        w -= sc.special.xlogy(1/beta, 1+beta) * step
        
        # Check convergence
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            return w

    return w


def exp_cf(item):
    w = np.zeros(items)
    mask = np.ones(items, dtype=bool)
    mask[item] = False
    w[mask]=irls_optimized(item,lam)
    return w


#ut.evaluate(np.eye(items),path)
lam = 200.0
W = Parallel(n_jobs=93)(delayed(exp_cf)(item) for item in tqdm(range(items)))
W = np.array(W)
ut.evaluate(W,path,ease=False)
ut.evaluate(W.T,path,ease=False)
np.save('slease' + path + '_200.npy',W)