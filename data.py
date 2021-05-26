import numpy as np


def random_graph(cf):
    # randomly generate an adjacency matrix (directed)
    adj = np.random.randint(2, size=[cf.input_size, cf.input_size])
    # add transpose and mod 2 to make it undirected
    adj = (adj.transpose() + adj)//2
    # omit self loops
    np.fill_diagonal(adj, 0)
    return adj


def load_data(cf):
    # if input specified, load that
    if cf.input_data:
        print('not implemented yet')
        return random_graph(cf)
    # otherwise, generate random graph
    else:
        return random_graph(cf)