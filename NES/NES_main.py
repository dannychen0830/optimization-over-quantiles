import netket as nk
import networkx as nx
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from NES.NES_energy import MIS_energy


# run NES using netket
def run_netket(cf, data, seed):
    # build objective
    if cf.pb_type == "maxindp":
        hamiltonian,graph,hilbert = MIS_energy(cf, data)
    if cf.pb_type == "maxcut":
        print('max cut not implemented yet')
        hamiltonian, graph, hilbert = MIS_energy(cf, data)

    # build model
    if cf.model_name == "rbm":
        model = nk.machine.RbmSpin(alpha=cf.width, hilbert=hilbert)
    elif cf.model_name == "rbm_real":
        model = nk.machine.RbmSpinReal(alpha=cf.width, hilbert=hilbert)
    model.init_random_parameters(seed=seed, sigma=cf.param_init)
    sampler = nk.sampler.MetropolisLocal(machine=model)

    # build optimizer
    if cf.optimizer == "adadelta":
        op = nk.optimizer.AdaDelta()
    elif cf.optimizer == "adagrad":
        op = nk.optimizer.AdaGrad(learning_rate=cf.learning_rate)
    elif cf.optimizer == "adamax":
        op = nk.optimizer.AdaMax(alpha=cf.learning_rate)
    elif cf.optimizer == "momentum":
        op = nk.optimizer.Momentum(learning_rate=cf.learning_rate)
    elif cf.optimizer == "rmsprop":
        op = nk.optimizer.RmsProp(learning_rate=cf.learning_rate)
    elif cf.optimizer == "sgd":
        op = nk.optimizer.Sgd(learning_rate=cf.learning_rate, decay_factor=cf.decay_factor)

    if cf.use_sr:
        method = "Sr"
    else:
        method = "Gd"

    # build algorithm
    gs = nk.variational.Vmc(
        hamiltonian=hamiltonian,
        sampler=sampler,
        method=method,
        optimizer=op,
        n_samples=cf.batch_size,
        use_iterative=cf.use_iterative,
        use_cholesky=cf.use_cholesky,
        diag_shift=0.1)

    # run algorithm
    start_time = time.time()
    gs.run(out='result', n_iter=cf.num_of_iterations, save_params_every=cf.num_of_iterations, show_progress=False)
    end_time = time.time()

    # plot the final node assignment if specified
    MIS_size = 0
    gen_sample = sampler.generate_samples(n_samples=1)
    assignment = np.zeros(gen_sample.shape[2])
    for i in range(gen_sample.shape[2]):
        assignment[i] = sum(gen_sample[0, :, i]) / gen_sample.shape[1]

    G = nx.from_numpy_matrix(data)
    pos = nx.circular_layout(G)
    color = []
    for i in range(cf.input_size):
        if assignment[i] > 0:
            MIS_size += 1
            color.append('red')
        else:
            color.append('blue')

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
    return MIS_size, time_elapsed, assignment
