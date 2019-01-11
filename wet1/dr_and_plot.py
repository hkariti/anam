import numpy as np
import matplotlib.pyplot as plt
import pca_functions
import manifold_functions
import datasets

from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def visualize_scatter_with_images(ax, reduced_data, images, image_zoom=0.5, num_of_images=80):
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1])
    artists = []
    rand_images_idx = np.random.randint(0, images.shape[1], num_of_images)
    images = images.T.reshape(-1, int(np.sqrt(images.shape[0])), int(np.sqrt(images.shape[0])))
    for xy, i in zip(reduced_data[rand_images_idx], images[rand_images_idx]):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom, cmap='gray')
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(reduced_data)
    ax.autoscale()


def pca_and_plot(data, data_name, labels=None):

    pca_reduced = pca_functions.pca(data)
    kpca_reduced = pca_functions.kpca(data)
    is_swiss_roll = labels is not None
    if is_swiss_roll:
        manual_reduced = pca_functions.manual_pca(data)
        subplots = 3
    else:
        subplots = 2

    fig = plt.figure()
    fig.suptitle("2D Representation of {}".format(data_name), fontsize=16)
    ax = plt.subplot(1, subplots, 1)
    if is_swiss_roll:
        ax.scatter(pca_reduced[:, 0], pca_reduced[:, 1], c=labels)
    else:
        visualize_scatter_with_images(ax, pca_reduced, data)
    ax.set_title("Linear PCA")

    ax = plt.subplot(1, subplots, 2)
    if is_swiss_roll:
        ax.scatter(kpca_reduced[:, 0], kpca_reduced[:, 1], c=labels)
    else:
        visualize_scatter_with_images(ax, kpca_reduced, data)
    ax.set_title("KPCA with RBF Kernel")

    if is_swiss_roll:
        ax = plt.subplot(1, subplots, 3)
        ax.scatter(manual_reduced[:, 0], manual_reduced[:, 1], c=labels)
        ax.set_title("PCA with Manual Features")
    plt.show()

    if is_swiss_roll:
        return {"PCA": pca_reduced, "KPCA": kpca_reduced, "Manual": manual_reduced}

    return {"PCA": pca_reduced, "KPCA": kpca_reduced}


def manifold_and_plot(data, data_name, labels=None):
    isomap_reduced = manifold_functions.isomap(data, neighbors=20)
    lle_reduced = manifold_functions.lle(data, neighbors=30)
    is_swiss_roll = labels is not None

    fig = plt.figure()
    fig.suptitle("2D Representation of {} Using Manifold Methods".format(data_name), fontsize=16)
    ax = plt.subplot(1, 2, 1)
    if is_swiss_roll:
        ax.scatter(isomap_reduced[:, 0], isomap_reduced[:, 1], c=labels)
    else:
        visualize_scatter_with_images(ax, isomap_reduced, data)
    ax.set_title("Isomap")

    ax = plt.subplot(1, 2, 2)
    ax.scatter(lle_reduced[:, 0], lle_reduced[:, 1], c=labels)
    if is_swiss_roll:
        ax.scatter(lle_reduced[:, 0], lle_reduced[:, 1], c=labels)
    else:
        visualize_scatter_with_images(ax, lle_reduced, data)
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
