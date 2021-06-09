import numpy as np
import networkx as nx
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from config import get_config
from data import load_data
# from src.util.helper import record_result ** we will see if we need this

from NES.NES_main import run_netket
from RNN.RNN_main import run_RNN
from KaMIS import run_KaMIS


# main function, runs the corresponding algorithm by directing to the right folder
def main(cf, seed):
    # create random graph as data (reserve the possibility of importing data)
    data = load_data(cf, seed)

    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))

    if cf.framework in ["NES"]:
        exp_name, score, time_elapsed = run_netket(cf, data, seed)
    if cf.framework in ["RNN"]:
        exp_name, score, time_elapsed = run_RNN(cf, seed)
    else:
        raise Exception("unknown framework")
    return exp_name, score, time_elapsed, bound


if __name__ == '__main__':
    "For single runs"

    # # parse the command line argument
    # cf, unparsed = get_config()
    # print(cf)
    #
    # # repeat multiple times if requested with
    # for num_trials in range(cf.num_trials):
    #     seed = cf.random_seed + num_trials
    #     np.random.seed(seed)
    #     tf.compat.v1.random.set_random_seed(seed)
    #     random.seed(seed)
    #
    #     exp_name, score, time_elapsed, bound = main(cf, seed)
    # print('finished')

    "For multiple runs and comparisons"

    # varying size
    min_size = 13
    d_size = 1
    max_size = 20
    num = int((max_size - min_size)/d_size)

    # varying density
    min_p = 0
    dp = 0.1
    max_p = 0.9

    num_rep = 5
    seed = 666

    MIS_size = np.zeros(shape=[3,num])
    var_MIS_size = np.zeros(shape=[3,num])
    time_elpased = np.zeros(shape=[3,num])
    var_time_elapsed = np.zeros(shape=[3,num])

    small_count = 0
    big_count = 0

    for size in range(min_size, max_size, d_size):
    # for p in range(min_p, max_p, dp):

        set_size = np.zeros(shape=[3,num_rep])
        duration = np.zeros(shape=[3,num_rep])

        for rep in range(num_rep):
            seed = seed + small_count
            np.random.seed(seed)
            tf.compat.v1.random.set_random_seed(seed)
            random.seed(seed)
            data = nx.to_numpy_array(nx.gnp_random_graph(size, 0.3, seed))

            set_size[0,rep], duration[0,rep] = run_netket(data=data, num_of_iterations=300, batch_size=1024, seed=seed, penalty=2)
            set_size[1,rep], duration[1,rep] = run_KaMIS(data)

            set_size[2,rep] = set_size[1,rep] - set_size[0,rep]
            duration[2, rep] = duration[1, rep] - duration[0, rep]

            small_count += 1

        for i in range(set_size.shape[0]):
            MIS_size[i,big_count] = np.mean(set_size[i,:])
            var_MIS_size[i,big_count] = np.var(set_size[i,:])
            time_elpased[i,big_count] = np.mean(duration[i,:])
            var_time_elapsed[i,big_count] = np.var(duration[i,:])

        big_count += 1

        print("Size " + str(size) + " done!")

    plt.figure(1)
    plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[0,:],color='b')
    plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[1,:],color='r')
    plt.xlabel('input graph size')
    plt.ylabel('independent set size')

    plt.figure(2)
    plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[2,:], yerr=var_MIS_size[2,:])
    plt.xlabel('input graph size')
    plt.ylabel('difference in set sizes')

    plt.figure(3)
    plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elpased[0, :], color='b')
    plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elpased[1, :], color='r')
    plt.xlabel('input graph size')
    plt.ylabel('average run time')

    plt.show()