import numpy as np
import networkx as nx


# if no data is specified, generate a random graph as a substitute
def random_graph(cf, seed):
    G = nx.gnp_random_graph(cf.input_size, 0.1, seed)
    return nx.to_numpy_array(G)


# either load data if available or generate a random graph
def load_data(cf, seed):
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
        return random_graph(cf, seed)