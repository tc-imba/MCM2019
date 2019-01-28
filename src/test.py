import numpy as np
from mayavi.mlab import *


def plot_fast():
    x = np.array([1])
    y = np.array([0])
    z = y

    phi = 0
    theta = phi

    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    w = np.cos(theta)

    obj = quiver3d(1,0,0, 0,0,1,
                   line_width=3, colormap='hsv',
                   scale_factor=0.8, mode='arrow', resolution=25)


plot_fast()
show()
