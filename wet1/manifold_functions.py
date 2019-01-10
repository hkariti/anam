import numpy as np
from sklearn.manifold import LocallyLinearEmbedding, Isomap

def lle(data, d=2, neighbors=5):
    lle_obj = LocallyLinearEmbedding(n_neighbors=neighbors, n_components=d)
    reduced = lle_obj.fit_transform(data.T)
    return reduced

def isomap(data, d=2, neighbors=5):
    isomap_obj = Isomap(n_neighbors=neighbors, n_components=d)
    reduced = isomap_obj.fit_transform(data.T)
    return reduced
