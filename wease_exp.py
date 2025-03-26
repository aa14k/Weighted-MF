import numpy as np
import utils as ut
import sys
from tqdm import tqdm
#import scipy as sc

def get_wease_V_cg(weight, lam, d, items, X, XtX_lam, V, U):
    v = V.flatten('F')
    pd = np.linalg.inv(U.T @ (XtX_lam) @ U)
    XU = X@U
    UtU = U.T @ U
    VUX = V@XU.T
    VUX[nonzerosT] *= weight 
    r = (W2X.T @ XU).flatten('F') - (lam*V@UtU + VUX@XU).flatten('F')
    z = (r.reshape((items,d),order='F')@pd).flatten('F')
    p = z.copy()
    for _ in (range(50)):
        P = p.reshape((items,d),order='F')
        PUX = P@XU.T
        PUX[nonzerosT] *= weight
        Ap = (lam * P@UtU + PUX @ XU).flatten('F')
        rtz = np.dot(r,z)
        alpha = rtz / np.dot(p,Ap)
        v += alpha * p
        r -= alpha * Ap
        #print('V:', np.linalg.norm(r_new))
        if np.linalg.norm(r) < 1e-8:
            return v.reshape((items,d),order='F')
        r = (z.reshape((items,d),order='F')@pd).flatten('F')
        beta = np.dot(r,z) / rtz
        p = z + beta * p
    return v.reshape((items,d),order='F')

def get_wease_U(weight, lam, d, items, X, XtX_inv, V, U, W2X):
    u = U.flatten('F')
    VtV = V.T @ V
    VtV_inv = np.linalg.inv(VtV)
    b = (X.T @ (W2X @ V)).flatten('F')
    XUV = X@U@V.T
    XUV[nonzeros] = weight * XUV[nonzeros]
    r = b - ((X.T @ (XUV @ V)).flatten('F') + lam * (U @ VtV).flatten('F'))
    z = (XtX_inv @ r.reshape((items,d),order='F') @ VtV_inv).flatten('F')
    p = z
    for _ in (range(50)):
        XPV = X@p.reshape((items,d),order='F')@V.T
        XPV[nonzeros] = weight * XPV[nonzeros]
        Ap = (X.T @ (XPV @ V)).flatten('F') + lam * (p.reshape((items,d),order='F') @VtV).flatten('F')
        rtz = np.dot(r,z)
        alpha = rtz / np.dot(p,Ap)
        u += alpha * p
        r -= alpha * Ap
        z = (XtX_inv @ r.reshape((items,d),order='F') @ VtV_inv).flatten('F')
        #print('U:', np.linalg.norm(r_new))
        if np.linalg.norm(r) < 1e-8:
            return u.reshape((items,d),order='F')
        beta = np.dot(r,z) / rtz
        p = z + beta * p
    return u.reshape((items,d),order='F')


if __name__=='__main__': # expects argument: <weightidx> <lamidx> <pathidx>
    weights = [1,2,5,10,20]
    lams = [1e-4,1e-2,1,1e2,1e4,1e5]
    paths = ['/efs/users/hsteck/public/datasets/netflix_prize_data/pro_sg', '/efs/users/hsteck/public/datasets/msd_data/pro_sg', '/efs/users/hsteck/public/datasets/movielens20mio/pro_sg']
    dims = [1000,100,10]
    runs = 5
    weight = weights[int(sys.argv[1])]
    lam = lams[int(sys.argv[2])]
    path = paths[int(sys.argv[3])]

    if path == '/efs/users/hsteck/public/datasets/msd_data/pro_sg':
        if lam == 1e5:
            lam = 3e4
    
    X = ut.load_train_data(path)
    XtX = (X.T @ X).toarray()
    users = X.shape[0]
    items = X.shape[1]
    nonzeros = X.nonzero()
    nonzerosT = (X.T).nonzero()
    
    for d in dims:
        identity = lam * np.eye(d)
        XtX_lam = XtX + lam * np.eye(items)
        XtX_inv = np.linalg.inv(XtX_lam)
        W2X = weight * X
        print('dim =', d,', weight =', weight, ' and lam =', lam)
        U = np.random.normal(size = (items,d)) * 0.01
        V = np.random.normal(size = (items,d)) * 0.01
        for i in tqdm(range(runs)):
            U = get_wease_U(weight, lam, d, items, X, XtX_inv, V, U, W2X)
            U = np.squeeze(np.array(U))
            V = get_wease_V_cg(weight, lam, d, items, X, XtX_lam, V, U)
            V = np.squeeze(np.array(V))
            if i == runs-1:
                ut.evaluate(U@V.T, path)
                print('=================================')