import numpy as np
import random
import tensorflow as tf
from config import get_config
from data import load_data
# from src.util.helper import record_result ** we will see if we need this

from NES.NES_main import run_netket


def main(cf, seed):
    # create random graph as data (reserve the possibility of importing data)
    data = load_data(cf)

    bound = None
    # run with algorithm options
    print("*** Running {} ***".format(cf.framework))
    if cf.framework in ["NES"]:
        exp_name, score, time_elapsed = run_netket(cf, data, seed)
    else:
        raise Exception("unknown framework")
    return exp_name, score, time_elapsed, bound


if __name__ == '__main__':
    cf, unparsed = get_config()
    print(cf)
    for num_trials in range(cf.num_trials):
        seed = cf.random_seed + num_trials
        np.random.seed(seed)
        tf.compat.v1.random.set_random_seed(seed)
        random.seed(seed)

        exp_name, score, time_elapsed, bound = main(cf, seed)
        # record_result(cf, exp_name, score, time_elapsed, bound) *** we'll see if we need this
    print('finished')

