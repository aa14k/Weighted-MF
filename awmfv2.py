import numpy as np
import utils as ut
from tqdm import tqdm
import sys 

def get_V_cg(U, V, weight, X, eye, lam, XU):
    WXT = weight * X.T
    nonzeros = WXT.nonzero()
    XU = X@U
    UXtXU = XU.T @ XU + eye
    pc = np.linalg.inv(UXtXU)
    VUX = V@XU.T
    VUX[nonzeros] *= weight
    r = WXT @ XU - (lam * V + VUX@XU)
    z = r @ pc
    p = z.copy()
    for _ in (range(20)):
        PUX = p@XU.T
        PUX[nonzeros] *= weight
        Ap = (PUX @ XU + lam*p)
        rtz = np.dot(r.flatten('F'),z.flatten('F'))
        alpha = rtz / np.dot(p.flatten('F'), Ap.flatten('F'))
        V += alpha * p
        r -= alpha * Ap
        if np.linalg.norm(r.flatten()) < 1e-8:
            return V
        z = r @ pc
        beta = np.dot(r.flatten('F'),p.flatten('F')) / rtz
        p = z + beta * p
    return V


def get_wmf_U2(weight, lam, d, X, V, U, XtX, XtX_inv):
    nonzeros = X.nonzero()
    VtV_inv = np.linalg.inv(V.T@V + lam * np.eye(d))
    XUV = X @ U @ V.T
    XUV[nonzeros] *= weight
    r = (X.T @ (weight*X @ V)) - (X.T @ (XUV@V) + lam*XtX@U)
    z = XtX_inv @ r @ VtV_inv
    p = z.copy()
    for _ in (range(20)):
        XPV = X @ p @ V.T
        XPV[nonzeros] *= weight
        Ap = X.T @ (XPV @ V) + lam*XtX@p
        rtz = np.dot(r.flatten('F'), z.flatten('F'))
        alpha = rtz / np.dot(p.flatten('F'), Ap.flatten('F'))
        U += alpha * p
        r -= alpha * Ap
        z = XtX_inv @ r @ VtV_inv
        if np.linalg.norm(r.flatten()) < 1e-8:
            return U
        beta = np.dot(r.flatten('F'), z.flatten('F')) / rtz
        p = z + beta * p
    return U

lam = float(sys.argv[1])
weight = float(sys.argv[2]) # 1, 2, 5, 10, 20

paths = ['/efs/users/hsteck/public/datasets/msd_data/pro_sg',
         '/efs/users/hsteck/public/datasets/netflix_prize_data/pro_sg',
         '/efs/users/hsteck/public/datasets/movielens20mio/pro_sg']
inv_names = ['msd', 'nfp', 'ml']
dims = [1000,100,10]

for i in range(len(paths)):
    path = paths[i]
    X = ut.load_train_data(path)
    XtX_inv = np.load(f'eigen/pinv_{inv_names[i]}.npy')
    users, items = X.shape
    XtX = (X.T @ X).toarray()
    for d in dims:
        eye = lam * np.eye(d)
        U, V = np.random.randn(items, d) * 0.01, np.random.randn(items, d) * 0.01
        print(f'(lam: {lam}, d: {d}, weight: {weight}, path: {path})')
        for _ in tqdm(range(5)):
            U = get_wmf_U2(weight, lam, d, X, V, U, XtX, XtX_inv)
            XU = X@U
            V = get_V_cg(U, V, weight, X, eye, lam, XU)
        ut.evaluate(U@V.T, path)
            



            

