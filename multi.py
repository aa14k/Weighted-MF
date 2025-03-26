from tqdm import tqdm
from scipy.optimize import minimize
from scipy import linalg

import numpy as np
import scipy as sc
from scipy.special import softmax, log_softmax

import os
from scipy import sparse
import pandas as pd

import bottleneck as bn

import datetime

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
    #items = random.sample(range(20108),1000)
    train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))
    X=train_data[:, :I]
    print (X.shape)
    return [X.shape[0] , X]


I = 5000
U = 116677
userCnt , X=load_xtx_binary(I)
clicks = np.squeeze(np.asarray(sc.sparse.csr_matrix.sum(X,axis=1)))
X_ = X.toarray()


def line_search(B,grad):
    beta = 0.4
    t=1
    norm_grad = np.linalg.norm(grad)**2
    for _ in range(20):
        first = lcat2(B - t*grad)
        second = lcat2(B) - t/2*norm_grad
        #print(_,first,second)
        if first-0.1 <= second:
            return t
        else:
            t = beta * t
    return t

def ease(X, lam = 200.0):
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


def SoftMax(x):
    """
    Protected SoftMax function to avoid overflow due to
    exponentiating large numbers.
    """

    # --> Add a feature associated to the neglected class.
    #x = np.insert(x, x.shape[1], 0, axis=1)

    # --> Max-normalization to avoid overflow.
    x -= np.max(x)

    return softmax(x, axis=1)

def Log_SoftMax(x):
    """
    Protected SoftMax function to avoid overflow due to
    exponentiating large numbers.
    """

    # --> Add a feature associated to the neglected class.
    #x = np.insert(x, x.shape[1], 0, axis=1)

    # --> Max-normalization to avoid overflow.
    x -= np.max(x)

    return log_softmax(x, axis=1)


# def line_search(B,grad):
#     beta = 0.2
#     t=1
#     norm_grad = np.linalg.norm(grad)**2
#     for _ in range(12):
#         first = lcat2(B - t*grad)
#         second = lcat2(B) - t/2*norm_grad
#         #print(_,first,second)
#         if first-0.1 <= second:
#             return t
#         else:
#             t = beta * t
#     return t

def lcat2(B,lam=100):
    #B = B.reshape(I,I)
    return -1.0 * np.sum(np.sum(X.multiply(Log_SoftMax(X@B)))) + lam/2*np.linalg.norm(B)**2


def dlcat2(B,lam=100):
    #B = B.reshape(I,I)
    XB = X@B
    S =  clicks.reshape(-1,1) * SoftMax(XB)
    grad = ((S-X).T@X + lam*B)
    return np.asarray(grad)


def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'))


N_test = test_data_tr.shape[0]
idxlist_test = range(N_test)

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG


def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall


def evaluate(BB):
    #evaluate in batches
    print(datetime.datetime.now())

    #makeSparseFormat(BB, 0.0)


    batch_size_test=5000
    n100_list, r20_list, r50_list = [], [], []



    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        Xtest = test_data_tr[idxlist_test[st_idx:end_idx]]
        Xtest = Xtest[:,:I]
        print (str(st_idx)+' ... '+str(end_idx))
        if sparse.isspmatrix(Xtest):
            Xtest = Xtest.toarray()
        Xtest = Xtest.astype('float32')

        #pred_val = Xtest.dot(BB_excl)
        #pred_val = (((Xtest-mu) * scaling).dot(BBth) / scaling) +mu   # no bias
        #pred_val = Xtest.dot(beta_0d)  # no bias
        #pred_val =Xtest.dot(beta_lowrank)  
        pred_val = Xtest.dot(BB)

        # exclude examples from training and validation (if any)
        pred_val[Xtest.nonzero()] = -np.inf
        n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
        r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))
        #calc_coverageCounts(coverageCounts2, pred_val)
        #break  # do only 5000 users

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

    print(datetime.datetime.now())
    return [np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)]

W_lin = ease(X,0.1)
evaluate(W_lin)



B = np.zeros((I,I))
diagIndices = np.diag_indices(B.shape[0])
for _ in tqdm(range(1000)):
    grad = dlcat2(B)
    #grad_ = grad
    grad[diagIndices] = 0.0
    #alpha = line_search(B,grad)
    B = B - 1e-5 * grad
    B[diagIndices] = -0.0
    #grad_ = grad
    #grad_[diagIndices] = 0.0
    evaluate(B)
    #evaluate(B.T)

# B = sol.x.reshape(I,I)
# np.save('multi_weights',B)
# evaluate(B.T)
# evaluate(B)






