import numpy as np
import networkx as nx
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from config import get_config
from data import load_data
# from src.util.helper import record_result ** we will see if we need this

from NES.NES_main import run_netket
from NES.NES_main import run_netket_2
from RNN.RNN_main import run_RNN
from RNN.train_RNN import run_2DTFIM
from KaMIS import run_KaMIS


# main function, runs the corresponding algorithm by directing to the right folder
def main(cf, seed):
    # create random graph as data (reserve the possibility of importing data)
    data = load_data(cf, seed)

    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))

    if cf.framework == "NES":
        MIS_size, time_elapsed, assignment = run_netket(cf, data, seed)
        if not check_solution(data, (assignment + 1) / 2):
            MIS_size = 0
    elif cf.framework == "RNN":
        MIS_size, time_elapsed, assignment = run_RNN(cf, data, seed)
        if not check_solution(data, assignment):
            MIS_size = 0
    elif cf.framework == "KaMIS":
        MIS_size, time_elapsed = run_KaMIS(data)
    else:
        raise Exception("unknown framework")

    return MIS_size, time_elapsed


# given a graph (adjcency matrix) and independent set assignment, check if the assignment is valid
def check_solution(adj, assignment):
    for i in range(adj.shape[0]):
        if round(assignment[i]) == 1:
            for j in range(adj.shape[1]):
                if adj[i,j] == 1 and round(assignment[j]) == 1:
                    return False
    return True


def single_run():
    # parse the command line argument
    cf, unparsed = get_config()
    print(cf)

    # repeat multiple times if requested with
    for num_trials in range(cf.num_trials):
        seed = cf.random_seed + num_trials
        np.random.seed(seed)
        tf.compat.v1.random.set_random_seed(seed)
        random.seed(seed)

        exp_name, score, time_elapsed, bound = main(cf, seed)
    print('finished')


def multiple_run_size(overwrite=False):
    cf, unparsed = get_config()

    "For multiple runs and comparisons"
    # varying size
    min_size = 5
    d_size = 1
    max_size = 6
    num = int((max_size - min_size) / d_size)

    num_rep = 1
    seed = 666

    MIS_size = np.zeros(num)
    var_MIS_size = np.zeros(num)
    time_elpased = np.zeros(num)
    var_time_elapsed = np.zeros(num)

    small_count = 0
    big_count = 0

    for size in range(min_size, max_size, d_size):
        cf.input_size = size

        set_size = np.zeros(num_rep)
        duration = np.zeros(num_rep)

        for rep in range(num_rep):
            seed = seed + small_count
            np.random.seed(seed)
            tf.compat.v1.random.set_random_seed(seed)
            random.seed(seed)

            set_size[rep], duration[rep] = main(cf, seed)

            small_count += 1

        MIS_size[big_count] = np.mean(set_size)
        var_MIS_size[big_count] = np.var(set_size)
        time_elpased[big_count] = np.mean(duration)
        var_time_elapsed[big_count] = np.var(duration)

        big_count += 1

        print("Size " + str(size) + " done!")

    if not overwrite:
        append_file('./output/mean_size.npy', MIS_size)
        append_file('./output/var_size.npy', var_MIS_size)
        append_file('./output/mean_time.npy', time_elpased)
        append_file('./output/var_time.npy', var_time_elapsed)
    else:
        np.save('./output/mean_size.npy', MIS_size)
        np.save('./output/var_size.npy', var_MIS_size)
        np.save('./output/mean_time.npy', time_elpased)
        np.save('./output/var_time.npy', var_time_elapsed)


def append_file(file, data):
    temp = np.load(file)
    if temp.ndim == 1:
        np.save(file, np.concatenate(([temp], [data])))
    else:
        np.save(file, np.concatenate((temp, [data])))


def comparison_density(size):

    # varying density
    min_p = 0.4
    dp = 0.1
    max_p = 0.6
    num = int(np.ceil((max_p-min_p)/dp))

    num_rep = 1
    seed = 666

    MIS_size = np.zeros(shape=[3, num])
    var_MIS_size = np.zeros(shape=[3, num])
    time_elpased = np.zeros(shape=[3, num])
    var_time_elapsed = np.zeros(shape=[3, num])

    small_count = 0
    big_count = 0

    for p in np.arange(min_p,max_p,dp):

        set_size = np.zeros(shape=[3, num_rep])
        duration = np.zeros(shape=[3, num_rep])

        for rep in range(num_rep):
            seed = seed + small_count
            np.random.seed(seed)
            tf.compat.v1.random.set_random_seed(seed)
            random.seed(seed)
            data = nx.to_numpy_array(nx.gnp_random_graph(size, p, seed))

            # run NES
            set_size[0, rep], duration[0, rep], assignment = run_netket_2(data=data, num_of_iterations=300,
                                                                          batch_size=1024, seed=seed, penalty=2)

            if not check_solution(data, (assignment + 1) / 2):
                print('# 1 TASK INCOMPLETE')
                set_size[0, rep] = 0

            # run KaMIS
            # set_size[1,rep], duration[1,rep] = run_KaMIS(data)

            # run RNN
            dummy1, dummy2, duration[1, rep], assignment, set_size[1, rep] = run_2DTFIM(numsteps=300, systemsize_x=size,
                                                                                        systemsize_y=1,
                                                                                        Bx=2, num_units=100,
                                                                                        numsamples=500,
                                                                                        learningrate=1e-3, seed=seed,
                                                                                        Jz=data)

            if not check_solution(data, assignment):
                print('# 2 TASK INCOMPLETE')
                set_size[1, rep] = 0

            set_size[2, rep] = set_size[1, rep] - set_size[0, rep]
            duration[2, rep] = duration[1, rep] - duration[0, rep]

            small_count += 1

        for i in range(set_size.shape[0]):
            MIS_size[i, big_count] = np.mean(set_size[i, :])
            var_MIS_size[i, big_count] = np.var(set_size[i, :])
            time_elpased[i, big_count] = np.mean(duration[i, :])
            var_time_elapsed[i, big_count] = np.var(duration[i, :])

        big_count += 1

        print("Size " + str(size) + " done!")

    plt.figure(1)
    plt.errorbar(np.arange(start=min_p, stop=max_p, step=dp), MIS_size[0, :], color='b', label='netket')
    plt.errorbar(np.arange(start=min_p, stop=max_p, step=dp), MIS_size[1, :], color='r', label='RNN')
    plt.legend()
    plt.xlabel('graph density (connection probability)')
    plt.ylabel('independent set size')

    plt.figure(2)
    plt.errorbar(np.arange(start=min_p, stop=max_p, step=dp), MIS_size[2, :],
                 yerr=np.sqrt(var_MIS_size[2, :]))
    plt.xlabel('graph density (connection probability)')
    plt.ylabel('difference in set sizes')

    plt.figure(3)
    plt.errorbar(np.arange(start=min_p, stop=max_p, step=dp), time_elpased[0, :], color='b', label='netket')
    plt.errorbar(np.arange(start=min_p, stop=max_p, step=dp), time_elpased[1, :], color='r', label='RNN')
    plt.legend()
    plt.xlabel('graph density (connection probability)')
    plt.ylabel('average run time')

    plt.show()

if __name__ == '__main__':
    #single_run()

    multiple_run_size(overwrite=False)
    print(np.load('./output/mean_size.npy'))
    #comparison_density(5)






    # plt.figure(1)
    # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[0, :], color='b', label='netket')
    # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[1, :], color='r', label='RNN')
    # plt.legend()
    # plt.xlabel('input graph size')
    # plt.ylabel('independent set size')
    #
    # plt.figure(2)
    # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[2, :], yerr=np.sqrt(var_MIS_size[2, :]))
    # plt.xlabel('input graph size')
    # plt.ylabel('difference in set sizes')
    #
    # plt.figure(3)
    # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elpased[0, :], color='b', label='netket')
    # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elpased[1, :], color='r', label='RNN')
    # plt.legend()
    # plt.xlabel('input graph size')
    # plt.ylabel('average run time')
    #
    # plt.show()