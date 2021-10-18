import time

import numpy as np
import netket as nk


# define the MIS hamiltonian
def MIS_energy(cf, adj):
    # construct the MIS hamiltonian
    J = cf.penalty*adj - np.eye(adj.shape[0])

    # define tools:
    # size of the graph
    N = J.shape[0]
    # operators on each site
    sz = np.array([[1., 0.], [0., -1.]])
    id = np.array([[1., 0.], [0., 1.]])
    b = (id - sz) / 2

    # create graph
    t1 = time.time()
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if J[i, j] != 0.:
                edges.append([i, j])
    t2 = time.time()
    # print('edges:', t2-t1)
    # g = nk.graph.CustomGraph(edges)
    g = nk.graph.Graph(edges=edges)
    t1 = time.time()
    print('graph construction:', t1 - t2)


    # system with spin-1/2 particles
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    t2 = time.time()
    # print('hilbert: ', t2 - t1)
    # hi = nk.hilbert.CustomHilbert(local_states=[-1,1], N=cf.input_size)
    # ha = nk.operator.LocalOperator(hi, 0)

    # # construct the hamiltonian
    # for i in range(N):
    #     for j in range(N):
    #         if J[i, j] != 0.:
    #             ha += J[i, j] * nk.operator.LocalOperator(hi, [np.kron(b, b)], [[i, j]])

    L = cf.penalty*np.kron(b, b)
    # ha = nk.operator.GraphOperator(hilbert=hi, graph=g, site_ops=[-1*b], bond_ops=[L])
    t1 = time.time()
    # print('hamiltonian:', t1-t2)

    t = time.time()
    ha = nk.operator.Ising(h=0, hilbert=hi, J=cf.penalty, graph=g)
    print('ising', time.time() - t)
    return ha, g, hi


def Maxcut_energy(cf, adj):
    N = adj.shape[0]
    # operators on each site
    sz = np.array([[1., 0.], [0., -1.]])

    # create graph
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] != 0.:
                edges.append([i, j])
    # g = nk.graph.CustomGraph(edges)
    g = nk.graph.Graph(edges=edges)

    # system with spin-1/2 particles
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    ha = nk.operator.GraphOperator(hilbert=hi, graph=g, bond_ops=[np.kron(sz, sz)])
    return ha, g, hi


def Transverse_Ising_Energy(cf, adj):
    N = adj.shape[0]
    z = np.array([[1, 0],[0, -1]])
    x = np.array([[0, 1],[1, 0]])

    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j] != 0.:
                edges.append([i, j])

    g = nk.graph.Graph(edges=edges)

    # system with spin-1/2 particles
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    ha = nk.operator.GraphOperator(hilbert=hi, graph=g, site_ops=[-1*x], bond_ops=[np.kron(z,z)])
    return ha, g, hi
