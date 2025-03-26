import utils as ut
import scipy as sc
import numpy as np
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

path = '/efs/users/hsteck/public/datasets/msd_data/pro_sg'
X = ut.load_train_data(path)
#X =sc.sparse.random(10000,1000,density=0.01,format='csr')
users,items = X.shape[0],X.shape[1]

def get_U_cg(pc,VtV_lam,V,weight,U,X):
    WX = (weight-1)*X
    XWV = np.array((weight*X).dot(V))
    UVtV = np.array(U@VtV_lam + (WX.multiply(U@V.T))@V)
    r = XWV - UVtV
    z = r.dot(pc)
    p = z.copy()
    for _ in tqdm(range(50)):
        pVtV = np.array(p@VtV_lam + (WX.multiply(p@V.T))@V)
        rtz = np.inner(r.flatten('F'),z.flatten('F'))  
        alpha = rtz / max(1e-32,np.inner(p.flatten('F'),pVtV.flatten('F')))
        U += alpha * p
        r -= alpha * pVtV
        rnorm = np.linalg.norm(r.flatten('F'))
        if rnorm <= 1e-8:
            return U
        z = r.dot(pc)
        beta = np.inner(r.flatten('F'),z.flatten('F')) / rtz
        p = z + beta * p
    return U


def get_V_cg(pc,WXT,UtU_lam,U,weight,V):
    XWU = np.array((weight*X).T.dot(U))
    VUtU = np.array((WXT.multiply(V@U.T))@U + V@UtU_lam)
    r = XWU - VUtU
    z = r.dot(pc)
    p = z.copy()
    for _ in tqdm(range(50)):
        pVtV = np.array((WXT.multiply(p@U.T))@U + p@UtU_lam)
        rtz = np.inner(r.flatten('F'),z.flatten('F'))  
        alpha = rtz / max(1e-32,np.inner(p.flatten('F'),pVtV.flatten('F')))
        V += alpha * p
        r -= alpha * pVtV
        rnorm = np.linalg.norm(r.flatten('F'))
        if rnorm <= 1e-8:
            return V
        z = r.dot(pc)
        beta = np.inner(r.flatten('F'),z.flatten('F')) / rtz
        p = z + beta * p
    return U

d = 2000
lam = 100.0
eye = lam * np.eye(d)
weight = 2
V = np.random.randn(items,d) * 0.01
U = np.random.randn(users,d) * 0.01
WX = (weight-1)*X
WXT = WX.T
for _ in tqdm(range(5)):
    VtV_lam = V.T @ V + eye
    pc = np.linalg.inv(VtV_lam)
    U = get_U_cg(pc,VtV_lam,V,weight,U,X)
    UtU_lam = U.T@U + eye
    pc = np.linalg.inv(UtU_lam)
    V = get_V_cg(pc,WXT,UtU_lam,U,weight,V)

X_val = ut.load_tr_te_data(path,False)
VtV_lam = V.T @ V + eye
pc = np.linalg.inv(VtV_lam)
U_val = np.random.randn(X.shape[0],d)*0.01
U_val = get_U_cg(pc,VtV_lam,V,weight,U_val,X_Val)
print('Validation Metrics')
ut.evaluate_wmf(U_Val@V.T, path, False)

X_test = ut.load_tr_te_data(path,True)
VtV_lam = V.T @ V + eye
pc = np.linalg.inv(VtV_lam)
U_test = np.random.randn(X.shape[0],d)*0.01
U_test = get_U_cg(pc,VtV_lam,V,weight,U_test,X_test)
print('Test Metrics')
ut.evaluate_wmf(U_test@V.T, path, False)




    


