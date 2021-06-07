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
    edges = []
    for i in range(N):
        for j in range(i, N):
            if J[i, j] != 0.: edges.append([i, j])
    g = nk.graph.CustomGraph(edges)

    # system with spin-1/2 particles
    hi = nk.hilbert.Spin(s=0.5, graph=g)
    ha = nk.operator.LocalOperator(hi, 0)

    # construct the hamiltonian
    for i in range(N):
        for j in range(N):
            if J[i, j] != 0.:
                ha += J[i, j] * nk.operator.LocalOperator(hi, [np.kron(b, b)], [[i, j]])
    return ha, g, hi