import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time

from NES.NES_main import run_netket


def simple_local_search(cf, data):
    start_time = time.time()
    N = cf.expansion

    indp_set = np.zeros(data.shape[0])
    visited = np.zeros(data.shape[0])

    while not np.array_equal(visited,np.ones(data.shape[0])):
        # pick reference node out of unvisited nodes
        r = data.shape[0] - np.count_nonzero(visited)
        rr = np.random.randint(low=0, high=r)
        ptr = -1
        while rr > -1:
            ptr += 1
            if visited[ptr] == 0:
                rr -= 1

        n0 = ptr

        # find neighborhood of radius N-1 n0
        subgraph_list = [n0]
        ref = [n0]
        visited[n0] = 1
        for t in range(1, N):
            for v in ref:
                ref.remove(v)
                if v not in subgraph_list:
                    subgraph_list.append(v)
                    visited[v] = 1
                for i in range(data[v, :].shape[0]):
                    if data[v, i] == 1 and visited[i] == 0:
                        ref.append(i)
        for v in ref:
            visited[v] = 1

        ns = len(subgraph_list)
        if ns == 1:
            subgraph_assignment = [1]
        else:
            # construct adjacency matrix of subgraph
            subgraph = np.zeros(shape=[ns,ns])
            for i in range(ns):
                for j in range(i+1, ns):
                    subgraph[i,j] = data[subgraph_list[i],subgraph_list[j]]
                    subgraph[j,i] = subgraph[i,j]

            # run VMC
            dummy1, dummy2, subgraph_assignment = run_netket(cf, subgraph, 666)
            subgraph_assignment = np.round((subgraph_assignment + 1)/2)

        # flip only if the prior assignment is 0
        for i in range(ns):
            if indp_set[subgraph_list[i]] == 0:
                indp_set[subgraph_list[i]] = subgraph_assignment[i]

    size = 0
    G = nx.from_numpy_matrix(data)
    pos = nx.circular_layout(G)
    color = []
    for i in range(cf.input_size):
        if indp_set[i] == 1:
            color.append('red')
            size += 1
        else:
            color.append('blue')

    if cf.print_assignment:
        # draw it out
        nx.draw(G, pos=pos, node_color=color)
        plt.title("Node Assignment")
        plt.show()

    end_time = time.time()

    return size, indp_set, end_time-start_time
