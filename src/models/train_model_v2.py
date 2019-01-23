import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, find
from sklearn.metrics import calinski_harabaz_score
from scipy.spatial.distance import euclidean
from sklearn.decomposition import TruncatedSVD
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.utils import shuffle


# reading the file and generating the csr matrix
def csr_read(fname, ftype="csr", nidx=1):
  
    
    with open(fname) as f:
        lines = f.readlines()
    
    if ftype == "clu":
        p = lines[0].split()
        nrows = int(p[0])
        ncols = int(p[1])
        nnz = long(p[2])
        lines = lines[1:]
        assert(len(lines) == nrows)
    elif ftype == "csr":
        nrows = len(lines)
        ncols = 0 
        nnz = 0 
        for i in xrange(nrows):
            p = lines[i].split()
            if len(p) % 2 != 0:
                raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(p)))
            nnz += len(p)/2
            for j in xrange(0, len(p), 2): 
                cid = int(p[j]) - nidx
                if cid+1 > ncols:
                    ncols = cid+1
    else:
        raise ValueError("Invalid sparse matrix ftype '%s'." % ftype)
    val = np.zeros(nnz, dtype=np.float)
    ind = np.zeros(nnz, dtype=np.int)
    ptr = np.zeros(nrows+1, dtype=np.long)
    n = 0 
    for i in xrange(nrows):
        p = lines[i].split()
        for j in xrange(0, len(p), 2): 
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j+1])
            n += 1
        ptr[i+1] = n 
    
    assert(n == nnz)
    
    return csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.float)


#scales the matrix by frequency of the terms.
def csr_idf(mat, copy=False, **kargs):
   
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

    
#normalizing the csr matrix
def csr_l2normalize(mat, copy=False, **kargs):
   
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = float(1.0/np.sqrt(rsum))
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat





# computing the centroids for the clusters
def findCentroids(mat, idx, k=2):
    centroids = list()
    for i in range(1,k+1):
        indi = [j for j, x in enumerate(idx) if x == i]
        members = mat[indi,:]
        if (members.shape[0] > 1):
            centroids.append(members.toarray().mean(0))
    
    centroids_csr = csr_matrix(centroids)
    return centroids_csr


#finding the appropriate clusters
def findCluster(mat, centroids):
    idx = list()
    similarityMatrix = mat.dot(centroids.T)

    for i in range(similarityMatrix.shape[0]):
        row = similarityMatrix.getrow(i).toarray()[0].ravel()
        top_indices = row.argsort()[-1]
        idx.append(top_indices + 1)
    return idx

# normal kmeans algorithm
def kmeans(mat,matrix, index_list, k):
    
    init_centroids_index = index_list[:2]
    centroids = mat[[init_centroids_index[0],init_centroids_index[1]],:]
    for itr in range(25):
        print("Iteration " + str(itr) + "\n")
        idx = findCluster(matrix,centroids)
        centroids = findCentroids(matrix,idx)
    index_list1 = []
    index_list2 = []
    for i in range(len(idx)):
        if idx[i] == 1:
            index_list1.append(index_list[i])
        elif idx[i] == 2:
            index_list2.append(index_list[i])
    cluster1 = mat[index_list1,:]
    cluster2 = mat[index_list2,:]
    return index_list1, index_list2, cluster1, cluster2, centroids[0], centroids[1]


#bisecting k-means algorithm 
def bisect(mat, k):
    matrix = mat
    cluster_list = []
    index_list = []

    for i in range(mat.shape[0]):
        index_list.append(i)

    while len(cluster_list) < k:
        sse1 = 0
        sse2 = 0
        index_list1, index_list2, cluster1, cluster2, centroids1, centroids2 = kmeans(mat,matrix,index_list,2)
        for clusters in cluster1:
            sse1 += (euclidean(clusters.toarray(),centroids1.toarray()))**2
        for clusters in cluster2:
            sse2 += (euclidean(clusters.toarray(),centroids2.toarray()))**2    
        if sse1 < sse2:
            cluster_list.append(index_list1)
            index_list = index_list2
            matrix = cluster2
        else:
            cluster_list.append(index_list2)
            index_list = index_list1
            matrix = cluster1
    cluster_list.append(index_list)
    return cluster_list



# matrix = csr_read("train.dat")
# mat2 = csr_idf(matrix, copy=True)    # after idf 
# mat3 = csr_l2normalize(mat2, copy=True)  # idf and normalize

# output = [0]* matrix.shape[0]
# k=7
# svd = TruncatedSVD(n_components=500, n_iter=50, random_state=42,algorithm='arpack')
# csrnorm_trunc=svd.fit_transform(mat3)
# csrnorm_trunc= csr_matrix(csrnorm_trunc)
# result = bisect(csrnorm_trunc,k)

# for i in range(len(result)):
#     for j in range(len(result[i])):
#         output[result[i][j]] = i+1

# print("Accuracy Score: ")
# print(calinski_harabaz_score(matrix.toarray(),output))

# if(k==7):
#     f = open("result_2.dat", "w")
#     f.write("\n".join(map(lambda x: str(x), output)))
#     f.close()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
