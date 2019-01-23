import datasets
import numpy as np
import euclidean_distances_functions
import matplotlib.pyplot as plt
import dr_and_plot
import image_compression

plt.rcParams['figure.figsize'] = [20, 10]

swiss_roll, colors = datasets.generate_swiss_roll()
faces = datasets.load_face_data()
mnist = datasets.load_mnist_data()

# datasets.plot_swiss_roll(swiss_roll, colors)
# datasets.plot_face_data(faces)
# datasets.plot_mnist_data(mnist)

k_1 = 3
k_2 = 100

datasets_dict = {"Swiss Roll": (swiss_roll, colors), "Faces": faces, "MNIST": mnist}

for dataset_name in datasets_dict:
    print("WORKING ON DATASET:", dataset_name)
    labels = None
    include_manual = False
    dataset = datasets_dict[dataset_name]
    if type(dataset) == tuple and len(dataset) == 2:
        labels = dataset[1]
        dataset = dataset[0]

    # 1 PCA Based
    print("Doing DR using PCA-based methods")
    algos_reduction = dr_and_plot.pca_and_plot(dataset, dataset_name, labels)

    # 2 Manifold Based
    print("Doing DR using manifold-based methods")
    manifold_reduction = dr_and_plot.manifold_and_plot(dataset, dataset_name, labels)
    algos_reduction.update(manifold_reduction)

    # 3 Euclidean Distances
    print("Calculating euclidean distances metrics")
    for algo in algos_reduction:
        print("For dr alrogithm:", algo)
        f, s, rd, d = euclidean_distances_functions.find_nn_pair(dataset, algos_reduction[algo])
        print("Points {}, {} are NN and have distance {} in low d and distance {} in high d".format(f, s, rd, d))
        f, s, rd, d = euclidean_distances_functions.find_far_pair(dataset, algos_reduction[algo])
        print("Points {}, {} are far and have distance {} in low d and distance {} in high d".format(f, s, rd, d))

        c, mds = euclidean_distances_functions.calc_c_and_mds_mean_values(dataset, algos_reduction[algo])
        print("C Mean is {}, MDS mean is {}".format(c, mds))

    # 4 Intrinsic Dimension
    m = euclidean_distances_functions.get_intrinsic_dimension_and_plot(dataset, k_1, k_2)
    print("Intrinsic Dimension: %f" % m)

print("IMAGE COMPRESSION")
image_compression.find_best_d_by_quality()
print("finished")
