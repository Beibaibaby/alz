
import numpy as np

fixseed = True
n = 20
A = 5  # see supp. fig S16
if fixseed:
    np.random.seed(7)
lj = np.round(np.random.uniform(size=n)) * 2 - 1  # random choose 1 or -1 as lj
kc = 8/75
kj = np.array([(kc * np.cos( j * np.pi / n),kc * np.sin( j * np.pi / n)) for j in range(n)])
if fixseed:
    np.random.seed(5)
phij = np.random.uniform(0, 2*np.pi, n)

def z(x):
    xx = lj * kj.dot(x) + phij
    return A * (np.cos(xx).sum() + 1j * np.sin(xx).sum())
gridsize = (750,750)
grid = np.zeros(gridsize)
for i in range(gridsize[0]):
    for j in range(gridsize[1]):
        grid[i,j] = (np.angle(z(np.array([i,j])),True)+180)/2

import matplotlib.pyplot as plt
import seaborn as sns

swarm_plot = sns.heatmap(grid,cmap='gist_rainbow')
fig = swarm_plot.get_figure()
fig.savefig("out.png") 



#save the plot

