#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

swiss_roll = generate_swiss_roll()
plot_swiss_roll(*swiss_roll)
