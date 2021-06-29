import numpy as np
import networkx as nx
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import multiprocessing as mp
import functools

from config import get_config
from data import load_data
# from src.util.helper import record_result ** we will see if we need this

from NES.NES_main import run_netket
from RNN.RNN_main import run_RNN
from KaMIS.KaMIS import run_KaMIS
from local_search import simple_local_search


# main function, runs the corresponding algorithm by directing to the right folder
def main(cf, seed):
    # create random graph as data (reserve the possibility of importing data)
    # data, n, m = load_data(cf, seed)
    G, list = load_data(cf, seed)
    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))

    if cf.framework == "NES":
        MIS_size = 0
        time_elapsed = 0
        for sub in list:
            data = nx.to_numpy_array(G.subgraph(sub))
            if data.shape[0] == 1:
                MIS_size += 1
            else:
                subset, sub_time, assignment = run_netket(cf, data, seed)
                if check_solution(data, (assignment + 1) / 2):
                    MIS_size += subset
                    time_elapsed += sub_time
    elif cf.framework == "RNN":
        data = nx.to_numpy_array(G)
        MIS_size, time_elapsed, assignment = run_RNN(cf, data, seed)
        if not check_solution(data, assignment):
            MIS_size = 0
    elif cf.framework == "KaMIS":
        data = nx.to_numpy_array(G)
        MIS_size, time_elapsed = run_KaMIS(data)
    elif cf.framework == "sLS":
        data = nx.to_numpy_array(G)
        MIS_size, assignment, time_elapsed = simple_local_search(cf, data)
        if not check_solution(data, assignment):
            MIS_size = 0
    else:
        raise Exception("unknown framework")

    print(MIS_size)
    print(time_elapsed)
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

        score, time = main(cf, seed)
    print('finished')


def multiple_run_size(min_size, d_size, max_size, num_rep):
    cf, unparsed = get_config()
    overwrite = cf.overwrite
    is_benchmark = cf.benchmark

    "For multiple runs and comparisons"
    # varying size
    num = int((max_size - min_size) / d_size)

    seed = 666

    # if it is a normal run, then load benchmark
    if not is_benchmark:
        MIS_size = np.zeros(num)
        max = np.zeros(num)
        min = np.zeros(num)
        benchmark = np.load('./output/benchmark.npy')
    # if this is a benchmark run, then allocate space for benchmark
    else:
        MIS_size = np.ones(num)
        var_MIS_size = np.zeros(num)
        benchmark = np.zeros(shape=[num, num_rep])
    time_elpased = np.zeros(num)
    var_time_elapsed = np.zeros(num)

    small_count = 0
    big_count = 0

    for size in range(min_size, max_size, d_size):
        cf.input_size = size

        set_size = np.zeros(num_rep)
        duration = np.zeros(num_rep)

        for rep in range(num_rep):
            use_seed = seed + small_count
            np.random.seed(use_seed)
            tf.compat.v1.random.set_random_seed(use_seed)
            random.seed(use_seed)

            set_size[rep], duration[rep] = main(cf, use_seed)
            print("set size found " + str(set_size[rep]))
            print("as opposed to " + str(benchmark[big_count, rep]))

            small_count += 1

        if not is_benchmark:
            ratio = set_size/benchmark[big_count,:]
            MIS_size[big_count] = np.mean(ratio)
            max[big_count] = np.max(ratio)
            min[big_count] = np.min(ratio)
        else:
            benchmark[big_count,:] = set_size
        time_elpased[big_count] = np.mean(duration)
        var_time_elapsed[big_count] = np.var(duration)

        big_count += 1

        print("Size " + str(size) + " done!")


    if is_benchmark:
        np.save('./output/benchmark.npy', benchmark)
    else:
        if not overwrite:
            append_file('./output/mean_size.npy', MIS_size)
            append_file('./output/max.npy', max)
            append_file('./output/min.npy', min)
            append_file('./output/mean_time.npy', time_elpased)
            append_file('./output/var_time.npy', var_time_elapsed)
        else:
            np.save('./output/mean_size.npy', MIS_size)
            np.save('./output/max.npy', max)
            np.save('./output/min.npy', min)
            np.save('./output/mean_time.npy', time_elpased)
            np.save('./output/var_time.npy', var_time_elapsed)


def multiple_run_size_parallel(min_size, d_size, max_size, num_rep):
    cf, unparsed = get_config()
    overwrite = cf.overwrite
    is_benchmark = cf.benchmark

    "For multiple runs and comparisons"
    # varying size
    num = int((max_size - min_size) / d_size)

    seed = 666

    # if it is a normal run, then load benchmark
    if not is_benchmark:
        MIS_size = np.zeros(num)
        max = np.zeros(num)
        min = np.zeros(num)
        benchmark = np.load('./output/benchmark.npy')
    # if this is a benchmark run, then allocate space for benchmark
    else:
        MIS_size = np.ones(num)
        var_MIS_size = np.zeros(num)
        benchmark = np.zeros(shape=[num, num_rep])
    time_elpased = np.zeros(num)
    var_time_elapsed = np.zeros(num)

    param = []
    size = min_size
    rep = 0
    for i in range(num*num_rep):
        param.append([seed+i, size])
        rep += 1
        if rep == num_rep:
            rep = 0
            size += d_size

    pool = mp.Pool(processes=num*num_rep)
    func = functools.partial(run_for_parallel, cf=cf)
    set, time = zip(*pool.map(func, param))

    ptr = 0
    for big_count in range(num):
        set_size = np.zeros(num_rep)
        for small_count in range(num_rep):
            set_size[small_count] = set[ptr]
            ptr += 1
        ratio = set_size/benchmark[big_count,:]
        MIS_size[big_count] = np.mean(ratio)
        max[big_count] = np.max(ratio)
        min[big_count] = np.min(ratio)

    if not overwrite:
        append_file('./output/mean_size.npy', MIS_size)
        append_file('./output/max.npy', max)
        append_file('./output/min.npy', min)
        # append_file('./output/mean_time.npy', time_elpased)
        # append_file('./output/var_time.npy', var_time_elapsed)
    else:
        np.save('./output/mean_size.npy', MIS_size)
        np.save('./output/max.npy', max)
        np.save('./output/min.npy', min)
        # np.save('./output/mean_time.npy', time_elpased)
        # np.save('./output/var_time.npy', var_time_elapsed)
    if is_benchmark:
            np.save('./output/benchmark.npy', benchmark)


def run_for_parallel(param, cf):
    seed = param[0]
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
    random.seed(seed)
    cf.input_size = param[1]

    set_size, duration = main(cf, seed)
    return set_size, duration


def append_file(file, data):
    temp = np.load(file)
    if temp.ndim == 1:
        np.save(file, np.concatenate(([temp], [data])))
    else:
        np.save(file, np.concatenate((temp, [data])))


if __name__ == '__main__':
    single_run()

    min_size = 5
    d_size = 1
    max_size = 13

    num_rep = 3

    # multiple_run_size_parallel(min_size, d_size, max_size, num_rep)
    # multiple_run_size(min_size, d_size, max_size, num_rep)
    # print(np.load('./output/mean_size.npy'))

    # MIS_size = np.load('./output/mean_size.npy')
    # min_MIS_size = np.load('./output/min.npy')
    # max_MIS_size = np.load('./output/max.npy')
    # # time_elapsed = np.load('./output/mean_time.npy')
    # # var_time_elapsed = np.load('./output/var_size.npy')
    # #
    # #
    # plt.figure(1)
    # # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[0, :],
    # #              yerr=np.sqrt(var_MIS_size[0, :]), color='b', label='KaMIS')
    # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[0,:],
    #              lolims=min_MIS_size[0,:], uplims=max_MIS_size[0,:], color='r', label='netket')
    # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[1, :],
    #              lolims=min_MIS_size[1, :], uplims=max_MIS_size[1, :], color='g', label='netket')
    # # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), MIS_size[3, :],
    # #              yerr=np.sqrt(var_MIS_size[3, :]), color='m', label='netket (500 iter)')
    # plt.legend()
    # plt.xlabel('input graph size')
    # plt.ylabel('approximation ratio')
    #
    # plt.show()
    #
    # plt.figure(2)
    # # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elapsed[0, :],
    # #              yerr=np.sqrt(var_time_elapsed[0, :]), color='b', label='KaMIS')
    # # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elapsed[1, :],
    # #              yerr=np.sqrt(var_time_elapsed[1, :]), color='r', label='netket')
    # # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elapsed[2, :],
    # #              yerr=np.sqrt(var_time_elapsed[2, :]), color='g', label='netket (500 iter)')
    # # plt.errorbar(np.arange(start=min_size, stop=max_size, step=d_size), time_elapsed[3, :],
    # #              yerr=np.sqrt(var_time_elapsed[3, :]), color='m', label='netket (500 iter)')
    # plt.legend()
    # plt.xlabel('input graph size')
    # plt.ylabel('average run time')
    #
    # plt.show()
    # #
    # # benchmark = np.load('output/benchmark.npy')
    # # print(MIS_size)


    # cf, unparsed = get_config()
    # data, n, m = load_data(cf, 666)
    # ls_size, dummy1 = local_search(5, cf, data)
    # km_size, dummy2 = run_KaMIS(data)
    #
    # print(ls_size)
    # print(km_size)