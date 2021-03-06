import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def define_args_parser():
    parser = argparse.ArgumentParser(description='Benchmark settings.')
    return parser

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--learning_rate', '-l', default=0.05, type=float, help='The learning rate')
net_arg.add_argument('--kernel_size', '-k', type=int, default=4, help='The kernel size of each conv layer')
net_arg.add_argument('--depth', '-d', type=int, default=1, help='Num of layers before sum pooling')
net_arg.add_argument('--width', '-w', type=int, default=1, help='Num of output channels in each layer')
net_arg.add_argument('--activation', type=str, choices=["relu", "tanh"], default="tanh", help='The activation function')
net_arg.add_argument('--model_name', '-m', type=str, \
                     choices=["rbm","crbm"], \
                     default='rbm', help='Model architecture')
net_arg.add_argument('--param_init', type=float, default=0.01, help='Model parameter initialization')
net_arg.add_argument('--penalty', type=float, default=1, help='During MIS, choose weighting of penalty term')
net_arg.add_argument('--num_units', type=int, default=100, help='the number of units in an RNN layer')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--pb_type', type=str, choices=["maxcut", "maxindp","transising"], default="maxindp", help='The problem type')
data_arg.add_argument('--batch_size', '-b', type=int, default=128, help='The batch size in each iteration')
data_arg.add_argument('--input_size', type=int, default=10, help='Number of spins in the input')
data_arg.add_argument('--num_of_iterations', '-ni', type=int, default=0, help='Num of iterations to benchmark')
data_arg.add_argument('--connect_prob', type=float, default=0.1, help='when using a Erdos-Renyi random graph, specify the connection probability')

# Train
train_arg = add_argument_group('Training')
train_arg.add_argument('--use_cholesky', type=str2bool, default=True, help='use cholesky solver in SR')
train_arg.add_argument('--use_iterative', type=str2bool, default=True, help='use iterative solver in SR')
train_arg.add_argument('--epochs', type=int, default=200)
train_arg.add_argument('--optimizer',
                       choices=["adam","adagrad","momentum","rmsprop","sgd"], \
                       default="sgd", help='The optimizer for training')
train_arg.add_argument('--use_sr', type=str2bool, default=True, help='use stochastic reconfiguration for training')
train_arg.add_argument('--decay_factor', type=float, default=1.0, help='Training decay factor')
train_arg.add_argument('--cvar', type=int, default=100, help='the percent of lower tail used for graident computation')
train_arg.add_argument('--nchain', type=int, default=16, help='the number of chains for Metropolis-Hastings')

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--framework', '-fr', type=str, \
                      choices= ["NES", "RNN", "KaMIS", "Exact", "sLS", "GNN", "BM"], \
                      default='NES', help='Options for algorithms')
misc_arg.add_argument('--dir', type=str, default='')
misc_arg.add_argument('--num_gpu', type=int, default=0)
misc_arg.add_argument('--random_seed', '-r', type=int, default=666, help='Randomization seed')
misc_arg.add_argument('--present', type=str, default="boxplot")
misc_arg.add_argument('--input_data', type=str2bool, default=False, help='Indicate whether input data is needed')
misc_arg.add_argument('--energy_plot', type=str2bool, default=False, help='Show the plot of energy vs. iteration')
misc_arg.add_argument('--print_assignment', type=str2bool, default=False, help='Show the final node assignment')
misc_arg.add_argument('--save_file', type=str, help='file name for where to save file')
misc_arg.add_argument('--expansion', type=int, default=2, help='during local search, how big is the expanded neighborhood')

def get_config():
    cf, unparsed = parser.parse_known_args()
    if cf.num_of_iterations == 0:
        cf.num_of_iterations = int(50 + 10*cf.batch_size/1024)
    return cf, unparsed
