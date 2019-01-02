#!/usr/bin/env python
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as sio

def generate_swiss_roll(size=3000):
    u = np.random.random(size)*10
    v = np.random.random(size)*10

    x = u*np.cos(u)
    y = u*np.sin(u)
    z = v
    color = u/u.max()

    return (x, y, z, color)

def plot_swiss_roll(x, y, z, color):
    for elav, azim in ((50, 40), (90, 0), (50, 90)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elav, azim)
        ax.scatter(x, y, z, c=color)
        plt.show()

def load_face_data(filename='./face_data.mat'):
    data = sio.loadmat(filename)
    return data['images']

def plot_face_data(images):
    indices = [1, 400, 532]
    for i in indices:
        fig = plt.figure()
        image = images[:, i].reshape(64,64).T
        plt.imshow(image, cmap='gray')
        plt.title('Face #{}'.format(i))
        plt.show()

def load_mnist_data(filename='./mnist.pkl.gz', digits=[0, 1, 2, 3], limit_per_digit=500):
    f = gzip.open(filename, 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
    f.close()

    train_set_images = train_set[0].T
    train_set_digit_number = train_set[1]
    
    dataset = np.ndarray((train_set_images.shape[0], len(digits) * limit_per_digit))
    for i, d in enumerate(digits):
        start_index = limit_per_digit * i
        end_index = limit_per_digit * (i + 1)
        dataset[:, start_index:end_index] = train_set_images[:, train_set_digit_number==d][:, :limit_per_digit]

    return dataset

def plot_mnist_data(mnist):
    indices = (50, 600, 1040)
    for i in indices:
        fig = plt.figure()
        image = mnist[:, i].reshape(28,28)
        plt.imshow(image, cmap='gray')
        plt.title('Image #{}'.format(i))
        plt.show()

if __name__ == '__main__':
    swiss_roll = generate_swiss_roll()
    plot_swiss_roll(*swiss_roll)

    images = load_face_data()
    plot_face_data(images)

    mnist = load_mnist_data()
    plot_mnist_data(mnist)
