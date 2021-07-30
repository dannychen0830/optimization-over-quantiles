import netket as nk
import networkx as nx
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from NES.NES_energy import MIS_energy
from NES.NES_energy import Maxcut_energy


# run NES using netket
def run_netket(cf, data, seed):
    # build objective
    if cf.pb_type == "maxindp":
        hamiltonian,graph,hilbert = MIS_energy(cf, data)
    if cf.pb_type == "maxcut":
        hamiltonian, graph, hilbert = Maxcut_energy(cf, data)

    # build model
    if cf.model_name == "rbm":
        model = nk.models.RBM(alpha=cf.width)
    elif cf.model_name == "crbm":
        model = nk.models.RBM(alpha=cf.width, dtype=np.complex64)
    # model.init_random_parameters(seed=seed, sigma=cf.param_init)
    sampler = nk.sampler.MetropolisLocal(hilbert=hilbert)

    # build optimizer
    if cf.optimizer == "adam":
        op = nk.optimizer.Adam(learning_rate=cf.learning_rate)
    elif cf.optimizer == "adagrad":
        op = nk.optimizer.AdaGrad(learning_rate=cf.learning_rate)
    elif cf.optimizer == "momentum":
        op = nk.optimizer.Momentum(learning_rate=cf.learning_rate)
    elif cf.optimizer == "rmsprop":
        op = nk.optimizer.RmsProp(learning_rate=cf.learning_rate)
    elif cf.optimizer == "sgd":
        op = nk.optimizer.Sgd(learning_rate=cf.learning_rate)

    if cf.use_sr:
        sr = nk.optimizer.SR()
    else:
        sr = None

    vs = nk.vqs.MCState(sampler=sampler, model=model, n_samples=cf.batch_size)
    gs = nk.VMC(hamiltonian=hamiltonian, optimizer=op, variational_state=vs, preconditioner=sr, alpha=cf.cvar)

    # run algorithm
    start_time = time.time()
    gs.run(out='result', n_iter=cf.num_of_iterations, save_params_every=cf.num_of_iterations, show_progress=True)
    end_time = time.time()

    # plot the final node assignment if specified
    size = 0
    assignment = gs.get_good_sample()

    G = nx.from_numpy_matrix(data)
    pos = nx.circular_layout(G)
    color = []
    for i in range(data.shape[0]):
        if assignment[i] > 0:
            if cf.pb_type == 'maxindp':
                size += 1
            color.append('red')
        else:
            color.append('blue')

    if cf.pb_type == 'maxcut':
        for i in range(data.shape[0]):
            for j in range(i+1, data.shape[0]):
                if data[i,j] == 1 and assignment[i] + assignment[j] == 0:
                    size += 1

    if cf.print_assignment:
        print(assignment)
        nx.draw(G, pos=pos, node_color=color)
        plt.title("Node Assignment")
        plt.show()

    # plot energy vs. iterations if specified
    if cf.energy_plot:
        file = json.load(open("result.log"))
        output = file["Output"]
        energy_data = np.zeros(len(output))
        var_data = np.zeros(len(output))
        for i in range(len(output)):
            energy_data[i] = output[i]["Energy"]["Mean"]
            var_data[i] = output[i]["Energy"]["Variance"]

        plt.errorbar(np.arange(len(output)), energy_data, yerr=np.sqrt(var_data), ecolor='tab:blue', color='r')
        plt.title("Energy per Iteration")
        plt.xlabel('number of iterations')
        plt.ylabel('mean energy')
        plt.show()

        # output result

    time_elapsed = end_time - start_time
    return size, time_elapsed, assignment
