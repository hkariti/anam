import numpy as np
import matplotlib.pyplot as plt
import pca_functions
import manifold_functions
import datasets


def pca_and_plot(data, data_name, labels=None, include_manual=False):

    pca_reduced = pca_functions.pca(data)
    kpca_reduced = pca_functions.kpca(data)
    if include_manual:
        manual_reduced = pca_functions.manual_pca(data)
        subplots = 3
    else:
        subplots = 2

    fig = plt.figure()
    fig.suptitle("2D Representation of {}".format(data_name), fontsize=16)
    ax = plt.subplot(1, subplots, 1)
    ax.scatter(pca_reduced[:, 0], pca_reduced[:, 1], c=labels)
    ax.set_title("Linear PCA")

    ax = plt.subplot(1, subplots, 2)
    ax.scatter(kpca_reduced[:, 0], kpca_reduced[:, 1], c=labels)
    ax.set_title("KPCA with RBF Kernel")

    if include_manual:
        ax = plt.subplot(1, subplots, 3)
        ax.scatter(manual_reduced[:, 0], manual_reduced[:, 1], c=labels)
        ax.set_title("PCA with Manual Features")
    plt.show()

    if include_manual:
        return {"PCA": pca_reduced, "KPCA": kpca_reduced, "Manual": manual_reduced}
    return {"PCA": pca_reduced, "KPCA": kpca_reduced}


def manifold_and_plot(data, data_name, labels=None):
    isomap_reduced = manifold_functions.isomap(data, neighbors=20)
    lle_reduced = manifold_functions.lle(data, neighbors=30)

    fig = plt.figure()
    fig.suptitle("2D Representation of {} Using Manifold Methods".format(data_name), fontsize=16)
    ax = plt.subplot(1, 2, 1)
    ax.scatter(isomap_reduced[:, 0], isomap_reduced[:, 1], c=labels)
    ax.set_title("Isomap")

    ax = plt.subplot(1, 2, 2)
    ax.scatter(lle_reduced[:, 0], lle_reduced[:, 1], c=labels)
    ax.set_title("LLE")
    plt.show()

    return {"Isomap": isomap_reduced, "LLE": lle_reduced}


def do_swiss_roll():
    swiss_roll, colors = datasets.generate_swiss_roll()
    pca_and_plot(swiss_roll, 'Swiss Roll', colors)
    manifold_and_plot(swiss_roll, 'Swiss Roll', colors)


if __name__ == '__main__':
    plt.rcParams['figure.figsize'] = [20, 10]
    do_swiss_roll()
