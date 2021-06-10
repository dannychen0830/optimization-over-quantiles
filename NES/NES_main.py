import netket as nk
import networkx as nx
import time
import json
import numpy as np
import matplotlib.pyplot as plt

from NES.NES_energy import MIS_energy
from NES.NES_energy import MIS_energy_2


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
    gs.run(out='result', n_iter=cf.num_of_iterations, save_params_every=cf.num_of_iterations)
    end_time = time.time()
    result = gs.get_observable_stats()

    # plot the final node assignment if specified
    if cf.print_assignment:
        gen_sample = sampler.generate_samples(n_samples=1)
        assignment = np.zeros(gen_sample.shape[2])
        for i in range(gen_sample.shape[2]):
            assignment[i] = sum(gen_sample[0,:,i])/gen_sample.shape[1]

        print(assignment)

        G = nx.from_numpy_matrix(data)
        pos = nx.circular_layout(G)
        color = []
        for i in range(cf.input_size):
            if assignment[i] > 0:
                color.append('red')
            else:
                color.append('blue')
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
    score = -result['Energy'].mean.real
    time_elapsed = end_time - start_time
    exp_name = cf.framework + str(cf.input_size)
    return exp_name, score, time_elapsed


def run_netket_2(data, num_of_iterations, batch_size, seed, pb_type="maxindp", model_name="rbm", optimizer="sgd",
               use_sr=True, width=1, param_init=0.01, learning_rate=0.01, decay_factor=1, use_iterative=True,
               use_cholesky=True, penalty=1,energy_plot=False, print_assignment=False):

    input_size = data.shape[0]

    # build objective
    if pb_type == "maxindp":
        hamiltonian,graph,hilbert = MIS_energy_2(data, penalty)
    if pb_type == "maxcut":
        print('max cut not implemented yet')
        hamiltonian, graph, hilbert = MIS_energy(data, penalty)

    # build model
    if model_name == "rbm":
        model = nk.machine.RbmSpin(alpha=width, hilbert=hilbert)
    elif model_name == "rbm_real":
        model = nk.machine.RbmSpinReal(alpha=width, hilbert=hilbert)
    model.init_random_parameters(seed=seed, sigma=param_init)
    sampler = nk.sampler.MetropolisLocal(machine=model)

    # build optimizer
    if optimizer == "adadelta":
        op = nk.optimizer.AdaDelta()
    elif optimizer == "adagrad":
        op = nk.optimizer.AdaGrad(learning_rate=learning_rate)
    elif optimizer == "adamax":
        op = nk.optimizer.AdaMax(alpha=learning_rate)
    elif optimizer == "momentum":
        op = nk.optimizer.Momentum(learning_rate=learning_rate)
    elif optimizer == "rmsprop":
        op = nk.optimizer.RmsProp(learning_rate=learning_rate)
    elif optimizer == "sgd":
        op = nk.optimizer.Sgd(learning_rate=learning_rate, decay_factor=decay_factor)

    if use_sr:
        method = "Sr"
    else:
        method = "Gd"

    # build algorithm
    gs = nk.variational.Vmc(
        hamiltonian=hamiltonian,
        sampler=sampler,
        method=method,
        optimizer=op,
        n_samples=batch_size,
        use_iterative=use_iterative,
        use_cholesky=use_cholesky,
        diag_shift=0.1)

    # run algorithm
    start_time = time.time()
    gs.run(out='result', n_iter=num_of_iterations, save_params_every=num_of_iterations, show_progress=False)
    end_time = time.time()
    result = gs.get_observable_stats()

    # plot the final node assignment if specified
    MIS_size = 0
    gen_sample = sampler.generate_samples(n_samples=1)
    assignment = np.zeros(gen_sample.shape[2])
    for i in range(gen_sample.shape[2]):
        assignment[i] = sum(gen_sample[0,:,i])/gen_sample.shape[1]

    G = nx.from_numpy_matrix(data)
    pos = nx.circular_layout(G)
    color = []
    for i in range(input_size):
        if assignment[i] > 0:
            MIS_size += 1
            color.append('red')
        else:
            color.append('blue')

    if print_assignment:
        print(assignment)
        nx.draw(G, pos=pos, node_color=color)
        plt.title("Node Assignment")
        plt.show()

    # plot energy vs. iterations if specified
    if energy_plot:
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