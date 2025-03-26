import os
import numpy as np
from scipy import sparse
import datetime
from scipy.special import expit as sigmoid 
import pandas as pd
import bottleneck as bn


def save_pkl(obj, filename ):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL )
    
def load_pkl(filename ):
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
#DATA_DIR = '/efs/users/hsteck/public/data_for_ease/movielens20mio/'
#DATA_DIR = '/efs/users/dliang/public/data/netflix_prize_data/download'
#pro_dir = '/efs/users/hsteck/public/datasets/netflix_prize_data/pro_sg'

def get_n_items(path):
    unique_sid = list()
    with open(os.path.join(path, 'unique_sid.txt'), 'r') as f:
        for line in f:
            unique_sid.append(line.strip())
    return len(unique_sid)

def load_train_data(path):
    tp = pd.read_csv(os.path.join(path, 'train.csv'))
    n_users = tp['uid'].max() + 1
    n_items = get_n_items(path)
    rows, cols = tp['uid'], tp['sid']
    return sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float32',
                             shape=(n_users, n_items))


def load_xtx_binary_val(): # deprecated
    #items = random.sample(range(20108),1000)
    return test_data_te


def load_tr_te_data(path,test=False):
    if test==False:
        tp_tr = pd.read_csv(os.path.join(path, 'validation_tr.csv'))
        tp_te = pd.read_csv(os.path.join(path, 'validation_te.csv'))
    else:
        tp_tr = pd.read_csv(os.path.join(path, 'test_tr.csv'))
        tp_te = pd.read_csv(os.path.join(path, 'test_te.csv'))
        
    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    n_items = get_n_items(path)

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float32', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float32', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te

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



def NDCG_binary_at_k_batch2(X_pred, heldout_batch, k=100):
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


def Recall_at_k_batch2(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall

def evaluate2(BB, test_data_tr, test_data_te): # deprecated
    print("evaluating ...")
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)

    batch_size_test = 5000
    n100_list, r20_list, r50_list = [], [], []
    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')

        pred_val = X.dot(BB)
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        n100_list.append(NDCG_binary_at_k_batch2(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
        r20_list.append(Recall_at_k_batch2(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch2(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))



def evaluate(BB,path,ease=True):
    print('Validation Metrics')
    evaluate_(BB,path,ease,test=False)
    print('Test Metrics')
    evaluate_(BB,path,ease,test=True)


def evaluate_WMF(BB,path):
    print('Validation Metrics')
    evaluate_wmf(BB,path)
    print('Test Metrics')
    evaluate_wmf(BB,path)


def evaluate_(BB, path, ease=True,test=False):
    test_data_tr, test_data_te = load_tr_te_data(path,test)
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)
    #evaluate in batches

    batch_size_test=5000
    n100_list, r20_list, r50_list = [], [], []



    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        Xtest = test_data_tr[idxlist_test[st_idx:end_idx]]
        if sparse.isspmatrix(Xtest):
            Xtest = Xtest.toarray()
        Xtest = Xtest.astype('float32')

        if ease==True:
            pred_val = Xtest.dot(BB)
        else:
            pred_val = sigmoid((2*Xtest-1).dot(BB))  

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

    if test==True:
        print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
        print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
        print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
    else:
        print("Val NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
        print("Val Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
        print("Val Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

    #print(datetime.datetime.now())
    return [np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)]



def evaluate_wmf(pred_vals,path,test): # deprecated
    #evaluate in batches
    test_data_tr, test_data_te = load_tr_te_data(path,test)
    batch_size_test=5000
    n100_list, r20_list, r50_list = [], [], []



    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        pred_val = pred_vals[st_idx:end_idx]
        Xtest = test_data_tr[idxlist_test[st_idx:end_idx]]
        print (str(st_idx)+' ... '+str(end_idx))
        if sparse.isspmatrix(Xtest):
            Xtest = Xtest.toarray()
        Xtest = Xtest.astype('float32')

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