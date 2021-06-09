import numpy as np
import matplotlib.pyplot as plt

from RNN.train_RNN import run_2DTFIM


#numsteps = number of training iterations
#systemsize_x = the size of the x-dimension of the square lattice
#systemsize_x = the size of the y-dimension of the square lattice
#Bx = transverse magnetic field
#numsamples = number of samples used for training
#num_units = number of memory units of the hidden state of the RNN
#num_layers is not supported yet, stay tuned!

# RNNEnergy, varRNNEnergy = run_2DTFIM(numsteps=1000, systemsize_x=30, systemsize_y=1, Bx=+2, num_units=150, numsamples=500,
#                learningrate=5e-4, seed=666)

#RNNEnergy is a numpy array of the variational energy of the pRNN wavefunction
#varRNNEnergy is a numpy array of the variance of the variational energy of the pRNN wavefunction

# numsteps = 2*10**4

def run_RNN(cf, seed):
    RNNEnergy, varRNNEnergy, time_elapsed = run_2DTFIM(numsteps=cf.num_of_iterations, systemsize_x=cf.input_size,
                                                       systemsize_y=1, Bx=cf.penalty, num_units=cf.num_units,
                                                       numsamples=cf.batch_size, learningrate=cf.learning_rate,
                                                       seed=seed, print_assignment=cf.print_assignment)

    if cf.energy_plot:
        plt.errorbar(np.arange(cf.num_of_iterations+1), RNNEnergy, yerr=np.sqrt(varRNNEnergy), ecolor='tab:blue', color='r')
        plt.title("Energy per Iteration")
        plt.xlabel('number of iterations')
        plt.ylabel('mean energy')
        plt.show()

    print("Time Elapsed:" + str(time_elapsed))
    return cf.framework + str(cf.input_size), RNNEnergy[cf.num_of_iterations], time_elapsed
