import numpy as np
from sklearn.decomposition import PCA, KernelPCA

def pca(data, d=2):
    pca_obj = PCA(d)
    reduced = pca_obj.fit_transform(data.T)
    return reduced

def kpca(data, d=2):
    norm_array = []
    for x in data.T:
        norm = np.linalg.norm(x)
        norm_array.append(norm)
    sigma = np.sqrt(np.var(np.array(norm_array)))
    kpca_obj = KernelPCA(d, kernel='rbf', gamma=sigma)
    reduced = kpca_obj.fit_transform(data.T)
    return reduced

def manual_pca(data, d=2):
    for i in range(data.shape[0]):
        for j in range(i, data.shape[0]):
            data=np.vstack((data, np.sqrt(data[j,:]**2 + data[i,:]**2)))
    return pca(data, d)
