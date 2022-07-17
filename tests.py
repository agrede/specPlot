import numpy as np
import specPlot.plotspec as p


xmnmxs = np.array([
    [-1.1, -1.001],
    [1.001, 1.1],
    [-1.2, -1.0],
    [1.0, 1.2],
    [-0.2, 0.],
    [0., 0.2],
    [-0.1, 0.001],
    [-0.001, 0.1]
    ])

ymnmxs = np.array([
    [-1.2, -0.9],
    [0.9, 1.2],
    [-1.3, -0.9],
    [0.9, 1.3],
    [-0.3, 0.1],
    [-0.1, 0.3],
    [-0.2, 0.2],
    [-0.2, 0.2]
    ])

step = 0.1
eps = np.finfo(type(step)).resolution

for k, xmnmx, ymnmx in zip(range(xmnmxs.shape[0]), xmnmxs, ymnmxs):
    tmnmx = np.array(list(p.data_minmax(xmnmx[0], xmnmx[1], step, False)))
    tpass = str(np.all(np.abs(ymnmx-tmnmx) < eps))
    print(f"{k:02d} - {tpass}: " +
          ", ".join([f"{x:0.3f}->{t:0.3f}" for x, t in zip(xmnmx, tmnmx)]))


ymnmxs = np.array([
    [-1.1, -1.1],
    [1.1, 1.1],
    [-1.2, -1.0],
    [1.0, 1.2],
    [-0.2, 0.0],
    [0., 0.2],
    [-0.1, 0.],
    [0., 0.1]
    ])

for k, xmnmx, ymnmx in zip(range(xmnmxs.shape[0]), xmnmxs, ymnmxs):
    tmnmx = np.array(list(p.data_minmax(xmnmx[0], xmnmx[1], step, True)))
    tpass = str(np.all(np.abs(ymnmx-tmnmx) < eps))
    print(f"{k:02d} - {tpass}: " +
          ", ".join([f"{x:0.3f}->{t:0.3f}" for x, t in zip(xmnmx, tmnmx)]))
