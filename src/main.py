import matplotlib.pyplot as plt
import matplotlib; matplotlib.interactive(True)

from azure_ml import submit_job
import torch

job_name = submit_job()


height = 28
width = 28

Q = network.nodes[0]
Ys = network.nodes[1:]

w = torch.stack([y.cpt.cpu() for y in Ys])

plt.figure()
for i in range(0, 10):
    plt.subplot(4, 3, i+1)
    plt.imshow(w[:, i, 1].reshape(height, width))
    plt.colorbar()
    plt.clim(0, 1)

pass