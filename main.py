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
from local_search import simple_local_search_parallel
from Exact.Exact import run_exact
# from GNN.GCN_main import run_GCN
# from BM import run_manopt


# main function, runs the corresponding algorithm by directing to the right folder
def main(cf, seed):
    # create random graph as data (reserve the possibility of importing data)
    # data, n, m = load_data(cf, seed)
    if cf.input_data:
        data = load_data(cf, seed)
    else:
        G, list = load_data(cf, seed)
    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))

    if cf.framework == "NES":
        MIS_size = 0
        time_elapsed = 0
        if not cf.input_data:
            for sub in list:
                data = nx.to_numpy_array(G.subgraph(sub))
                if data.shape[0] == 1:
                    if cf.pb_type == 'maxindp':
                        MIS_size += 1
                else:
                    subset, sub_time, assignment = run_netket(cf, data, seed)
                    MIS_size += subset
                    time_elapsed += sub_time
                    if cf.pb_type == 'maxindp' and not check_solution(data, (assignment + 1) / 2):
                        MIS_size = 0
        else:
            MIS_size, time_elapsed, assignment = run_netket(cf, data, seed)
            if not check_solution(data, (assignment + 1) / 2):
                MIS_size = 0
                time_elapsed = 0
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
    elif cf.framework == "Exact":
        data = nx.to_numpy_array(G)
        MIS_size, time_elapsed = run_exact(data)
    # elif cf.framework == "GNN":
    #     MIS_size = 0
    #     time_elapsed = 0
    #     for sub in list:
    #         data = nx.to_numpy_array(G.subgraph(sub))
    #         if data.shape[0] == 1:
    #             MIS_size += 1
    #         else:
    #             subset, sub_time, assignment = run_GCN(cf, data)
    #             if check_solution(data, assignment):
    #                 MIS_size += subset
    #                 time_elapsed += sub_time
    # elif cf.framework == 'BM':
    #     data = nx.to_numpy_array(G)
    #     MIS_size, time_elapsed = run_manopt(cf, data)
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
    seed = cf.random_seed
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

    size = min_size
    ptr = 0
    for big_count in range(num):
        param = []
        for small_count in range(num_rep):
            param.append([seed+ptr, size])
            ptr += 1

        pool = mp.Pool(processes=num_rep)
        func = functools.partial(submain_size, cf=cf)
        set, time = zip(*pool.map(func, param))

        MIS_size[big_count,:] = set
        time_elapsed[big_count,:] = time

        size += d_size

    np.save('./output/'+cf.save_file + "_size", MIS_size)
    np.save('./output/'+cf.save_file + "_time", time_elapsed)


def compare_batch_size(min_batch, d_batch, max_batch, num_rep):
    cf, unparsed = get_config()
    num = int((max_batch - min_batch) / d_batch)

    size_n = np.zeros(shape=[num, num_rep])
    time_n = np.zeros(shape=[num, num_rep])

    seed = 666

    cf.batch_size = min_batch
    for i in range(num):
        seed_list = []
        for j in range(num_rep):
            seed_list.append(seed+j)
        pool = mp.Pool(processes=num_rep)
        func = functools.partial(submain_batch, cf=cf)
        size_n[i,:], time_n[i,:] = zip(*pool.map(func, seed_list))
        cf.batch_size += d_batch

    np.save('./output/compare_' + cf.save_file + "_size", size_n)
    np.save('./output/compare_' + cf.save_file + "_time", time_n)


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
    single_run()

    min_size = 10
    d_size = 5
    max_size = 70

    num_rep = 10

    # multiple_run_size_parallel(min_size, d_size, max_size, num_rep)
    # multiple_run_size(min_size, d_size, max_size, num_rep)
    # compare_batch_size(2000, 2000, 12000, 5)
    # print(np.load('./output/mean_size.npy'))

    # file = open('list.txt','r')
    # a = np.load('./output/compare_maxcut_cont_size.npy')
    # b = np.load('./output/compare_maxcut_cont_time.npy')
    # s = np.zeros(shape=[4,10])
    # t = np.zeros(shape=[4,10])
    # i = 0
    # j = 0
    # b = True
    # for line in file:
    #     if line[0].isdigit():
    #         if b:
    #             s[i,j] = int(line)
    #             b = not b
    #         else:
    #             t[i,j] = float(line)
    #             b = not b
    #             j += 1
    #             if j == 10:
    #                 j = 0
    #                 i += 1
    # s[3,:] = a
    # t[3,:] = b
    # print(s)
    # print(t)
    # np.save('./output/compare_maxcut_size.npy', s)
    # np.save('./output/compare_maxcut_time.npy', t)

    # s_k = np.concatenate((np.load('./output/KaMIS_size.npy'),np.load('./output/KaMIS_cont_size.npy')), axis=0)[0:17,:]
    # t_k = np.concatenate((np.load('./output/KaMIS_time.npy'),np.load('./output/KaMIS_cont_time.npy')), axis=0)[0:17,:]
    # s_n = np.concatenate((np.load('./output/netket_reg_4000_size.npy'),np.load('./output/netket_reg_4000_cont_size.npy')), axis=0)[0:17,:]
    # t_n = np.concatenate((np.load('./output/netket_reg_4000_time.npy'),np.load('./output/netket_reg_4000_cont_time.npy')), axis=0)[0:17,:]
    # s_c = np.concatenate((np.load('./output/netket_crbm_4000_size.npy'), np.load('./output/netket_crbm_4000_cont_size.npy')), axis=0)[0:17,:]
    # t_c = np.concatenate((np.load('./output/netket_crbm_4000_time.npy'), np.load('./output/netket_crbm_4000_cont_time.npy')), axis=0)[0:17,:]
    # s_g = np.concatenate((np.load('./output/GNN_369_5_size.npy'), np.zeros(shape=[6,10])), axis=0)
    # t_g = np.concatenate((np.load('./output/GNN_369_5_time.npy'), np.zeros(shape=[6,10])), axis=0)
    #
    #
    s_axis = np.concatenate((np.arange(start=min_size, stop=max_size, step=d_size), np.arange(start=70, stop=250, step=30)))
    #
    # s_axis = np.arange(start=min_size, stop=max_size, step=d_size)
    s_k = np.concatenate((np.load('./output/netket_CVar_10_size.npy'), np.load('./output/netket_CVar_10_s_size.npy')), axis=0)
    t_k = np.concatenate((np.load('./output/netket_CVar_10_time.npy'), np.load('./output/netket_CVar_10_s_time.npy')), axis=0)
    s_n = np.concatenate((np.load('./output/netket_CVar_25_size.npy'), np.load('./output/netket_CVar_25_s_size.npy')), axis=0)
    t_n = np.concatenate((np.load('./output/netket_CVar_25_time.npy'), np.load('./output/netket_CVar_25_s_time.npy')), axis=0)
    s_c = np.concatenate((np.load('./output/netket_CVar_100_size.npy'), np.load('./output/netket_CVar_100_s_size.npy')), axis=0)
    t_c = np.concatenate((np.load('./output/netket_CVar_100_time.npy'), np.load('./output/netket_CVar_100_s_time.npy')), axis=0)
    # s_n = np.load('./output/maxcut_CVar_10_size.npy')
    # t_n = np.load('./output/maxcut_CVar_10_time.npy')
    # s_c = np.load('./output/maxcut_CVar_25_size.npy')
    # t_c = np.load('./output/maxcut_CVar_25_time.npy')
    # s_g = np.load('./output/maxcut_CVar_100_size.npy')
    # t_g = np.load('./output/maxcut_CVar_100_time.npy')
    # #
    # plt.figure(1)
    # plt.errorbar(s_axis, np.mean(s_k, axis=1), yerr=[np.mean(s_k, axis=1)-np.min(s_k, axis=1), np.max(s_k, axis=1)-np.mean(s_k, axis=1)], color='b', label='alpha = 0.1')
    # plt.errorbar(s_axis, np.mean(s_n, axis=1), yerr=[np.mean(s_n, axis=1)-np.min(s_n, axis=1), np.max(s_n, axis=1)-np.mean(s_n, axis=1)], color='m', label='alpha = 0.25')
    # plt.errorbar(s_axis, np.mean(s_c, axis=1), yerr=[np.mean(s_c, axis=1)-np.min(s_c, axis=1), np.max(s_c, axis=1)-np.mean(s_c, axis=1)], color='r', label='alpha = 1')
    # # plt.errorbar(s_axis, np.mean(s_g, axis=1), yerr=[np.mean(s_g, axis=1)-np.min(s_g, axis=1), np.max(s_g, axis=1)-np.mean(s_g, axis=1)], color='c', label='alpha = 1')
    # plt.xlabel('number of vertices')
    # plt.ylabel('size of the indepedent set')
    # plt.legend()
    # #
    # plt.figure(3)
    # plt.yscale('log')
    # plt.errorbar(s_axis, np.mean(t_k, axis=1), yerr=np.std(t_k, axis=1), color='b', label='alpha = 0.1')
    # plt.errorbar(s_axis, np.mean(t_n, axis=1), yerr=np.std(t_n, axis=1), color='m', label='alpha = 0.25')
    # plt.errorbar(s_axis, np.mean(t_c, axis=1), yerr=np.std(t_c, axis=1), color='r', label='alpha = 1')
    # # plt.errorbar(s_axis, np.mean(t_g, axis=1), yerr=np.std(t_g, axis=1), color='c', label='alpha = 1')
    # plt.xlabel('number of vertices')
    # plt.ylabel('time used (log scale)')
    # plt.legend()

    # plt.figure(4)
    # plt.plot(s_axis, np.max(s_k, axis=1), color='b', label='KaMIS')
    # plt.plot(s_axis, np.max(s_n, axis=1), color='m', label='r-rbm')
    # plt.plot(s_axis, np.max(s_c, axis=1), color='r', label='c-rbm')
    # plt.plot(s_axis, np.max(s_g, axis=1), color='c', label='GNN')
    # plt.xlabel('number of vertices')
    # plt.ylabel('maximum independent set found')
    # plt.legend()

    # plt.show()

    # s = np.load('./output/compare_maxindp_v2_size.npy')
    # s = np.mean(s, axis=1)
    # t = np.load('./output/compare_maxindp_v2_time.npy')
    # t = np.mean(t, axis=1)
    # bm = np.load('./output/BM_250_size.npy')
    #
    # axis = np.array([0.01, 0.1, 0.25, 1])
    #
    # fig, ax1 = plt.subplots()
    # ax1.set_xscale('log')
    #
    # ax1.set_xlabel('alpha')
    # ax1.set_ylabel('independent set size found', color='k')
    # ax1.plot(axis, s)
    # # ax1.axhline(np.mean(bm), color='r')
    # ax1.tick_params(axis='y', labelcolor='k')
    #
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    #
    # ax2.set_ylabel('time', color='k')  # we already handled the x-label with ax1
    # ax2.plot(axis, t, linestyle='dashed')
    # ax2.tick_params(axis='y', labelcolor='k')
    #
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # fig.legend()
    # plt.show()