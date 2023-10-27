import matplotlib.pyplot as plt
import torch


def plot_tensor(tensor):
    mean = torch.mean(tensor)
    tensor = tensor.data.clone()
    tensor[tensor >= mean] = 255
    tensor[tensor < mean] = 0
    plt.imshow(tensor.cpu().numpy(), cmap="hot")
    plt.show()


def plot_filters(filters):
    filters = filters.clone()
    filters = filters.reshape(filters.shape[0], filters.shape[2], filters.shape[3])
    filters = filters[0:2]
    for tensor in filters:
        plot_tensor(tensor)

def analyze(model):
    filters = model.layer1[0].weight.data
    plot_filters(filters)