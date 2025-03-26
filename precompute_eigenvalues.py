import numpy as np
import utils as ut
import scipy as sc

#path = '/efs/users/hsteck/public/datasets/netflix_prize_data/pro_sg'
path = '/efs/users/hsteck/public/datasets/msd_data/pro_sg'
X= ut.load_train_data(path)
XtX = (X.T @ X).toarray()
print('Getting eigendecomposition of XtX')
xvals, xvecs = np.linalg.eigh(XtX)
np.save('eigen/msd-xvals.npy', xvals)
np.save('eigen/msd-xvecs.npy', xvecs)
