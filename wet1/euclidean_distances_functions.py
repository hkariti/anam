from sklearn.neighbors import NearestNeighbors
import numpy as np

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
