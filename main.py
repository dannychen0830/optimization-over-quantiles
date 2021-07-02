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
from local_search import simple_local_search_parallel
from Exact.Exact import run_exact


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
        MIS_size, assignment, time_elapsed = simple_local_search_parallel(cf, data)
        if not check_solution(data, assignment):
            print('failed')
            MIS_size = 0
    elif cf.frameework == "Exact":
        data = nx.to_numpy_array(G)
        MIS_size, time_elapsed = run_exact(data)
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

    "For multiple runs and comparisons"
    # varying size
    num = int((max_size - min_size) / d_size)

    seed = 666

    MIS_size = np.zeros(shape=[num, num_rep])
    time_elapsed = np.zeros(shape=[num, num_rep])

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
            # print("set size found " + str(set_size[rep]))

            small_count += 1

        MIS_size[big_count,:] = set_size
        time_elapsed[big_count,:] = duration

        big_count += 1

        # print("Size " + str(size) + " done!")

    np.save('./output/' + cf.save_file + "_size", MIS_size)
    np.save('./output/' + cf.save_file + "_time", time_elapsed)


def multiple_run_size_parallel(min_size, d_size, max_size, num_rep):
    cf, unparsed = get_config()

    "For multiple runs and comparisons"
    # varying size
    num = int((max_size - min_size) / d_size)

    seed = 666

    MIS_size = np.zeros(shape=[num, num_rep])
    time_elapsed = np.zeros(shape=[num, num_rep])

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
    func = functools.partial(submain_size, cf=cf)
    set, time = zip(*pool.map(func, param))

    ptr = 0
    for big_count in range(num):
        set_size = np.zeros(num_rep)
        sub_time = np.zeros(num_rep)
        for small_count in range(num_rep):
            set_size[small_count] = set[ptr]
            sub_time[small_count] = time[ptr]
            ptr += 1
        # ratio = set_size/benchmark[big_count,:]
        MIS_size[big_count,:] = set_size
        time_elapsed[big_count,:] = sub_time

    np.save('./output/'+cf.save_file + "_size", MIS_size)
    np.save('./output/'+cf.save_file + "_time", time_elapsed)


def compare_batch_size(min_batch, d_batch, max_batch, num_rep):
    cf, unparsed = get_config()
    num = int((max_batch - min_batch) / d_batch)

    size_b = np.zeros(num_rep)
    size_n = np.zeros(shape=[num, num_rep])
    time_b = np.zeros(num_rep)
    time_n = np.zeros(shape=[num, num_rep])

    seed = 666

    cf.framework = 'KaMIS'
    for i in range(num_rep):
        size_b[i], time_b[i] = main(cf, seed+i)

    cf.framework = 'NES'
    cf.batch_size = min_batch
    for i in range(num):
        seed_list = []
        for j in range(num_rep):
            seed_list.append(seed+j)
        pool = mp.Pool(processes=num_rep)
        func = functools.partial(submain_batch, cf=cf)
        size_n[i,:], time_n[i,:] = zip(*pool.map(func, seed_list))
        size_n[i,:] = np.divide(size_n[i,:],size_b)
        cf.batch_size += d_batch

    np.save('./output/compare_size', size_n)
    np.save('./output/compare_time', time_n)


def submain_size(param, cf):
    seed = param[0]
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
    random.seed(seed)
    cf.input_size = param[1]

    set_size, duration = main(cf, seed)
    return set_size, duration

def submain_batch(seed, cf):
    np.random.seed(seed)
    tf.compat.v1.random.set_random_seed(seed)
    random.seed(seed)

    return main(cf, seed)


def append_file(file, data):
    temp = np.load(file)
    if temp.ndim == 1:
        np.save(file, np.concatenate(([temp], [data])))
    else:
        np.save(file, np.concatenate((temp, [data])))


if __name__ == '__main__':
    # single_run()

    min_size = 10
    d_size = 5
    max_size = 70

    num_rep = 10

    # multiple_run_size_parallel(min_size, d_size, max_size, num_rep)
    # multiple_run_size(min_size, d_size, max_size, num_rep)
    # compare_batch_size(2000, 2000, 12000, 5)
    # print(np.load('./output/mean_size.npy'))

    # s_k = np.load('./output/KaMIS_size.npy')
    # t_k = np.load('./output/KaMIS_time.npy')
    # s_sp = np.load('./output/netket_sLs_p_size.npy')
    # t_sp = np.load('./output/netket_sLs_p_time.npy')
    # s_s = np.load('./output/netket_sLs_size.npy')
    # t_s = np.load('./output/netket_sLs_time.npy')
    # s_n = np.load('./output/netket_reg_size.npy')
    # t_n = np.load('./output/netket_reg_time.npy')
    #
    # s_axis = np.arange(start=min_size, stop=max_size, step=d_size)
    #
    # plt.figure(1)
    # plt.errorbar(s_axis, np.mean(s_k, axis=1), yerr=[np.mean(s_k, axis=1)-np.min(s_k, axis=1), np.max(s_k, axis=1)-np.mean(s_k, axis=1)], color='b', label='KaMIS')
    # plt.errorbar(s_axis, np.mean(s_s, axis=1), yerr=[np.mean(s_s, axis=1)-np.min(s_s, axis=1), np.max(s_s, axis=1)-np.mean(s_s, axis=1)], color='r', label='(mistake) local search')
    # plt.errorbar(s_axis, np.mean(s_sp, axis=1), yerr=[np.mean(s_sp, axis=1)-np.min(s_sp, axis=1), np.max(s_sp, axis=1)-np.mean(s_sp, axis=1)], color='g', label='local search')
    # plt.errorbar(s_axis, np.mean(s_n, axis=1), yerr=[np.mean(s_n, axis=1)-np.min(s_n, axis=1), np.max(s_n, axis=1)-np.mean(s_n, axis=1)], color='m', label='netket')
    # plt.xlabel('number of vertices')
    # plt.ylabel('size of the indepedent set')
    # plt.legend()
    #
    # r_s = np.divide(s_s, s_k)
    # r_n = np.divide(s_n, s_k)
    # r_sp = np.divide(s_sp, s_k)
    # plt.figure(2)
    # plt.errorbar(s_axis, np.mean(r_s, axis=1), yerr=[np.mean(r_s, axis=1)-np.min(r_s, axis=1), np.max(r_s, axis=1)-np.mean(r_s, axis=1)], color='r', label='(mistake) local search')
    # plt.errorbar(s_axis, np.mean(r_n, axis=1), yerr=[np.mean(r_n, axis=1)-np.min(r_n, axis=1), np.max(r_n, axis=1)-np.mean(r_n, axis=1)], color='m', label='netket')
    # plt.errorbar(s_axis, np.mean(r_sp, axis=1), yerr=[np.mean(r_sp, axis=1)-np.min(r_sp, axis=1), np.max(r_sp, axis=1)-np.mean(r_sp, axis=1)], color='g', label='local search')
    # plt.xlabel('number of vertices')
    # plt.ylabel('normalized set size')
    # plt.legend()
    #
    # plt.figure(3)
    # plt.yscale('log')
    # plt.errorbar(s_axis, np.mean(t_k, axis=1), yerr=np.std(t_k, axis=1), color='b', label='KaMIS')
    # plt.errorbar(s_axis, np.mean(t_s, axis=1),yerr=np.std(t_s, axis=1), color='r', label='(mistake) local search')
    # plt.errorbar(s_axis, np.mean(t_sp, axis=1), yerr=np.std(t_sp, axis=1), color='g', label='local search')
    # plt.errorbar(s_axis, np.mean(t_n, axis=1), yerr=np.std(t_n, axis=1), color='m', label='netket')
    # plt.xlabel('number of vertices')
    # plt.ylabel('time used (log scale)')
    # plt.legend()
    #
    # plt.figure(4)
    # plt.plot(s_axis, np.max(s_k, axis=1), color='b', label='KaMIS')
    # plt.plot(s_axis, np.max(s_s, axis=1), color='r', label='(mistake) local search')
    # plt.plot(s_axis, np.max(s_sp, axis=1), color='g', label='local search')
    # plt.plot(s_axis, np.max(s_n, axis=1), color='m', label='netket')
    # plt.xlabel('number of vertices')
    # plt.ylabel('maximum independent set found')
    # plt.legend()
    #
    #
    # plt.show()

    # s = np.load('./output/compare_size.npy')
    # t = np.load('./output/compare_time.npy')
    #
    # axis = np.arange(start=2000, stop=12000, step=2000)
    #
    # fig, ax1 = plt.subplots()
    #
    # color = 'tab:red'
    # ax1.set_xlabel('batch size')
    # ax1.set_ylabel('independent set size found', color=color)
    # ax1.errorbar(axis, np.mean(s,axis=1), yerr=[np.mean(s, axis=1)-np.min(s, axis=1), np.max(s, axis=1)-np.mean(s, axis=1)], color=color)
    # ax1.tick_params(axis='y', labelcolor=color)
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # color = 'tab:blue'
    # ax2.set_ylabel('time', color=color)  # we already handled the x-label with ax1
    # ax2.plot(axis, np.mean(t, axis=1), color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    #
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()