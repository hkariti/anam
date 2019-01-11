from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


# Q2.1.3 s1
def find_nn_pair(data, reduced_data):
    nn_obj = NearestNeighbors(n_neighbors=1)
    reduced_data_nn = nn_obj.fit(reduced_data)
    nn_distances, nn_idx = reduced_data_nn.kneighbors()

    first_point_idx = np.argmin(nn_distances)
    second_point_idx = nn_idx[first_point_idx][0]
    reduced_distance = nn_distances[first_point_idx][0]

    first_point = data[:, first_point_idx]
    second_point = data[:, second_point_idx]
    distance = np.linalg.norm(first_point - second_point)

    return first_point_idx, second_point_idx, reduced_distance, distance


# Q2.1.3 s2
def find_far_pair(data, reduced_data):
    nn_obj = NearestNeighbors(n_neighbors=40)
    reduced_data_nn = nn_obj.fit(reduced_data)
    nn_distances, nn_idx = reduced_data_nn.kneighbors()

    found = False
    for first_point_idx in range(nn_distances.shape[0]):
        for second_point_idx in range(nn_distances.shape[0]):
            if first_point_idx == second_point_idx:
                continue
            if second_point_idx not in nn_idx[first_point_idx] \
               and first_point_idx not in nn_idx[second_point_idx]:
                found = True
                break
        if found:
            break
    assert found

    first_reduced_point = reduced_data[first_point_idx]
    second_reduced_point = reduced_data[second_point_idx]
    reduced_distance = np.linalg.norm(first_reduced_point - second_reduced_point)
    first_point = data[:, first_point_idx]
    second_point = data[:, second_point_idx]
    distance = np.linalg.norm(first_point - second_point)

    return first_point_idx, second_point_idx, reduced_distance, distance


# Q2.1.3 s3-s4
def calc_c_and_mds_mean_values(data, reduced_data):
    nn_obj = NearestNeighbors(n_neighbors=12)
    nn_reduced_obj = NearestNeighbors(n_neighbors=12)

    reduced_data_nn = nn_reduced_obj.fit(reduced_data)
    data_nn = nn_obj.fit(data.T)

    nn_reduced_distances, nn_reduced_idx = reduced_data_nn.kneighbors()
    nn_distances, nn_idx = data_nn.kneighbors()

    # calc C function
    c_idx_list = []
    c_size_array = np.zeros(data.shape[1])
    for data_point_idx in range(data.shape[1]):
        mask = np.logical_not(np.in1d(nn_reduced_idx[data_point_idx], nn_idx[data_point_idx]))
        c_idx_list.append(nn_reduced_idx[data_point_idx][mask])
        c_size_array[data_point_idx] = mask.sum()

    c_mean = c_size_array.mean()

    mds = np.zeros(data.shape[1])
    for data_point_idx in range(data.shape[1]):
        if c_idx_list[data_point_idx].shape[0] == 0:
            continue
        mds_temp = reduced_data[c_idx_list[data_point_idx]] - reduced_data[data_point_idx]
        mds_temp = np.linalg.norm(mds_temp)
        mds_temp = mds_temp / c_idx_list[data_point_idx].shape[0]
        mds[data_point_idx] = mds_temp
    mds_mean = mds.mean()

    return c_mean, mds_mean


def get_intrinsic_dimension(data, k_1, k_2):

    num_of_samples = data.shape[1]
    # nn_dist_mat = np.zeros([data.shape[1], data.shape[1]])
    # for feature_idx in range(data.shape[0]):
    #     data_1_vec = data[feature_idx][:, None]
    #     nn_dist_mat += (data_1_vec - data_1_vec.T) ** 2
    # nn_dist_mat = np.sqrt(nn_dist_mat)
    nn_dist_mat = pdist(data.T)
    nn_dist_mat = squareform(nn_dist_mat)
    nn_idx_mat = np.argsort(nn_dist_mat)

    m = np.zeros(k_2 - k_1 + 1)
    t_mat = np.zeros([k_2+1, num_of_samples])
    for k in range(k_2+1):
        t_mat[k] = nn_dist_mat[np.arange(0, num_of_samples), nn_idx_mat[:, k]]

    for k in range(k_1, k_2+1):
        print(k)
        m_k = np.zeros(num_of_samples)
        for j in range(1, k-1):
            if np.any(t_mat[k] - t_mat[j] < 0):
                print("HEEEY")
            m_k += np.log(t_mat[k] / t_mat[j])
        m_k /= k-1
        m_k = m_k.mean()
        m[k-k_1] = 1 / m_k
    return m


def get_intrinsic_dimension_and_plot(data, k_1, k_2):
    m = get_intrinsic_dimension(data, k_1, k_2)
    fig = plt.figure()
    plt.title("$m_k$ as function of $k$", fontsize=16)
    plt.plot(np.arange(k_1, k_2 + 1), m)
    plt.show()
    return m.mean()
