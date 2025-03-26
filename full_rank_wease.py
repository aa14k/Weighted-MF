import utils as ut
import scipy as sc
import numpy as np
from tqdm import tqdm
import scipy.sparse as sparse
from joblib import Parallel, delayed


#path = '/efs/users/hsteck/public/datasets/msd_data/pro_sg'
# print('loading data')
# X = ut.load_train_data(path)
# #X = sc.sparse.random(10000,1000,0.01,format='csr')
# users,items = X.shape[0],X.shape[1]
# XtX = (X.T@X).toarray()
# nonzeros = X.nonzero()

def hadamard_matmal(r,X,B,cols,ind):
    idx = cols[ind[r]:ind[r+1]]
    return (B[idx][:,idx]).sum(0)

def wease_cg(pc, X, XtX_lam, weight, nonzeros, path, users, items):
    ind = X.indptr
    B = np.zeros((items,items))
    XWX = (X.T.dot(weight*X)).toarray() # 30 seconds
    temp = Parallel(n_jobs=36)(delayed(hadamard_matmal)(user,X,B,nonzeros[1],ind) for user in range(users)) # 2 minutes
    WXXB = (weight - 1) * sparse.csr_matrix((np.concatenate(temp), (nonzeros[0], nonzeros[1])))
    XtXB = np.array(X.T.dot(WXXB) + XtX_lam.dot(B)) # 1.5 minutes...
    r = XWX - XtXB
    z = pc.dot(r) # 1.5 minutes 
    p = z.copy()
    for _ in tqdm(range(20)):
        temp = Parallel(n_jobs=36)(delayed(hadamard_matmal)(user,X,p,nonzeros[1],ind) for user in range(users))
        WXXp = (weight - 1) * sparse.csr_matrix((np.concatenate(temp), (nonzeros[0], nonzeros[1])))
        XtXp = np.array(X.T.dot(WXXp) + XtX_lam.dot(p))
        rtz = np.inner(r.flatten('F'),z.flatten('F'))  
        alpha = rtz / max(1e-32,np.inner(p.flatten('F'),XtXp.flatten('F')))
        B += alpha * p
        r -= alpha * XtXp
        rnorm = np.linalg.norm(r.flatten('F'))
        if rnorm <= 1e-8:
            ut.evaluate(B,path)
            break
        if _ % 5 == 4:
            ut.evaluate(B,path)
        z = pc.dot(r)
        beta = np.inner(r.flatten('F'),z.flatten('F')) / rtz
        p = z + beta * p
    

#paths = ['/efs/users/hsteck/public/datasets/msd_data/pro_sg', '/efs/users/hsteck/public/datasets/movielens20mio/pro_sg']
paths = ['/efs/users/hsteck/public/datasets/msd_data/pro_sg']

lam = 1
weights = [1]
for path in paths:
    print(path)
    X = ut.load_train_data(path)
    users,items = X.shape[0],X.shape[1]
    XtX = (X.T@X).toarray()
    nonzeros = X.nonzero()
    XtX_lam = XtX.copy()
    XtX_lam[np.diag_indices(items)] += lam
    pc = np.linalg.inv(XtX_lam)
    for weight in weights:
        print('running with (lam,weight): ', lam,' , ', weight)
        wease_cg(pc, X, XtX_lam, weight, nonzeros, path, users, items)
