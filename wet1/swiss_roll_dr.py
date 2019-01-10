import numpy as np
import matplotlib.pyplot as plt
import pca_functions
import manifold_functions
import datasets

def swiss_pca_and_plot(data, labels=None):
    pca_reduced = pca_functions.pca(data)
    kpca_reduced = pca_functions.kpca(data)
    manual_reduced = pca_functions.manual_pca(data)

    fig = plt.figure()
    fig.suptitle("2D Representation of Swiss Roll", fontsize=16)
    ax = plt.subplot(1, 3, 1)
    ax.scatter(pca_reduced[:, 0], pca_reduced[:, 1], c=labels)
    ax.set_title("Linear PCA")

    ax = plt.subplot(1, 3, 2)
    ax.scatter(kpca_reduced[:, 0], kpca_reduced[:, 1], c=labels)
    ax.set_title("KPCA with RBF Kernel")

    ax = plt.subplot(1, 3, 3)
    ax.scatter(manual_reduced[:, 0], manual_reduced[:, 1], c=labels)
    ax.set_title("PCA with Manual Features")
    plt.show()

def swiss_manifold_and_plot(data, labels=None):
    isomap_reduced = manifold_functions.isomap(data, neighbors=20)
    lle_reduced = manifold_functions.lle(data, neighbors=30)

    fig = plt.figure()
    fig.suptitle("2D Representation of Swiss Roll Using Manifold Methods", fontsize=16)
    ax = plt.subplot(1, 2, 1)
    ax.scatter(isomap_reduced[:, 0], isomap_reduced[:, 1], c=labels)
    ax.set_title("Isomap")

    ax = plt.subplot(1, 2, 2)
    ax.scatter(lle_reduced[:, 0], lle_reduced[:, 1], c=labels)
    ax.set_title("LLE")
    plt.show()

if __name__ == '__main__':
    swiss_roll, colors = datasets.generate_swiss_roll()
    swiss_pca_and_plot(swiss_roll, colors)
    swiss_manifold_and_plot(swiss_roll, colors)
