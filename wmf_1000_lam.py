import numpy as np
import utils as ut
import sys
from tqdm import tqdm
import scipy as sc

def get_V_cg(U, V, WXT, weight, X, eye, lam):
    nonzeros = WXT.nonzero()
    XU = X@U
    UXtXU = XU.T @ XU + eye
    pc = np.linalg.inv(UXtXU)
    VUX = V@XU.T
    VUX[nonzeros] *= weight
    r = WXT @ XU - (lam * V + VUX@XU)
    z = r @ pc
    p = z.copy()
    for _ in (range(50)):
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


def get_wmf_U(weight, lam, d, items, X, xvals, xvecs, V, U):
    nonzeroU = X.nonzero()
    u = U.flatten('F')
    VtV = V.T @ V
    vvals, vvecs = np.linalg.eigh(VtV)
    D_inv = 1 / (np.kron(vvals,xvals) + lam)
    XUV = X @ U @ V.T # dimension mismatch
    XUV[nonzeroU] = weight * XUV[nonzeroU]
    r = (X.T @ (W2X @ V)).flatten('F') - (X.T @ (XUV@V) + lam*U).flatten('F')
    z = (xvecs @ (D_inv * (xvecs.T @ r.reshape((items,d),order='F') @ vvecs).flatten('F')).reshape((items,d),order='F') @ vvecs.T).flatten('F')
    p = z
    for _ in (range(50)):
        XPV = X @ p.reshape((items,d),order='F') @ V.T
        XPV[nonzeroU] = weight * XPV[nonzeroU]
        Ap = (X.T @ (XPV @ V)).flatten('F') + lam * p
        rtz = np.dot(r,z)
        alpha = rtz / np.dot(p,Ap)
        u = u + alpha * p
        r_new = r - alpha * Ap
        DXRV = D_inv * (xvecs.T @ r_new.reshape((items,d),order='F') @ vvecs).flatten('F')
        z = (xvecs @ DXRV.reshape((items,d),order='F') @ vvecs.T).flatten('F')
        if np.linalg.norm(r_new) < 1e-8:
            return u.reshape((items,d),order='F')
        beta = np.dot(r_new,z) / rtz
        p = z + beta * p
        r = r_new
    return u.reshape((items,d),order='F')

if __name__=='__main__': # expects argument: <weightidx> <lamidx> <pathidx>
    weights = [1,2,5]
    lams = [1e-9,1e-7,1e-5]
    runs = 5
    d=1000

    path = '/efs/users/hsteck/public/datasets/netflix_prize_data/pro_sg'
    eigenpath = 'np'
    

    X = ut.load_train_data(path)
    XtX = (X.T @ X).toarray()
    users, items = X.shape
    nonzeros = X.nonzero()
    nonzerosT = (X.T).nonzero()
    print('Loading eigendecomposition of XtX')
    xvals = np.load('eigen/' + eigenpath + '-xvals.npy')
    xvecs = np.load('eigen/' + eigenpath + '-xvecs.npy')

    for weight in weights:
        for lam in lams:
            eye = lam * np.eye(d)
            W2X = weight * X
            W2XT = W2X.T
            print('dim =', d,', weight =', weight, ' and lam =', lam)
            U = np.random.normal(size = (items,d)) * 0.01
            V = np.random.normal(size = (items,d)) * 0.01
            for i in tqdm(range(runs)):
                U = get_wmf_U(weight, lam, d, items, X, xvals, xvecs, V, U)
                U = np.squeeze(np.array(U))
                V = get_V_cg(U, V, W2X.T, weight, X, eye, lam)
                V = np.squeeze(np.array(V))
                if i == runs - 1:
                    ut.evaluate(U@V.T, path)
                    print("========================================")