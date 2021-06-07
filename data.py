import numpy as np


# if no data is specified, generate a random graph as a substitute
def random_graph(cf):
    # randomly generate an adjacency matrix (directed)
    adj = np.random.randint(2, size=[cf.input_size, cf.input_size])
    # add transpose and mod 2 to make it undirected
    adj = (adj.transpose() + adj)//2
    # omit self loops
    np.fill_diagonal(adj, 0)
    return adj


# either load data if available or generate a random graph
def load_data(cf):
    # if input specified, load that
    if cf.input_data:
        data = np.loadtxt("data.txt", dtype=int)
        if len(data.shape) > 2 or data.shape[0] != data.shape[1]:
            raise Exception("Invalid Data!")
        if np.count_nonzero(np.diagonal(data)) > 0:
            raise Exception("Self Loop!")
        if cf.input_size != data.shape[0]:
            print("Inconsistent input size!")
            cf.input_size = data.shape[0]
        return data
    # otherwise, generate random graph
    else:
        return random_graph(cf)