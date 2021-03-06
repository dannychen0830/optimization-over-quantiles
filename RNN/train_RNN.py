import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from math import ceil

from RNN.RNN_wave import RNNwavefunction
from RNN.MDRNNcell import MDRNNcell


tf.compat.v1.compat.v1.logging.set_verbosity(tf.compat.v1.compat.v1.logging.ERROR)  # stop displaying tensorflow warnings
tf.compat.v1.disable_v2_behavior()


# Loading Functions --------------------------
def Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, queue_samples, log_probs_tensor, samples_placeholder, log_probs,
                           sess):
    """ To get the local energies of 2D TFIM (OBC) given a set of set of samples in parallel!
    Returns: The local energies that correspond to the "samples"
    Inputs:
    - samples: (numsamples, Nx,Ny)
    - Jz: (Nx,Ny) np array *** changed: Jz is now the adjacency matrix
    - Bx: float
    - queue_samples: ((Nx*Ny+1)*numsamples, Nx,Ny) an empty allocated np array to store the non diagonal elements
    - log_probs_tensor: A TF tensor with size (None)
    - samples_placeholder: A TF placeholder to feed in a set of configurations
    - log_probs: ((Nx*Ny+1)*numsamples): an empty allocated np array to store the log_probs non diagonal elements
    - sess: The current TF session
    """

    numsamples = samples.shape[0]

    N = Nx * Ny  # Total number of spins

    local_energies = np.zeros((numsamples), dtype=np.float64)

    # assume linear samples for now
    for n in range(numsamples):
        for i in range(N-1):
            if samples[n,i,0] == 1:
                local_energies[n] -= 1 # try to maximize set, so reward if in the set
            # else:
            #     local_energies[n] += 1 # is this allowed? Penalizing for not being in the set?
            for j in range(N):
                if Jz[i,j] == 1 and samples[n,i,0] + samples[n,j,0] == 2:
                    local_energies[n] += Bx # adjacent nodes shouldn't be in the set, penalize!
                # else:
                #     local_energies[n] += Bx
    return local_energies


# --------------------------


# ---------------- Running VMC with 2DRNNs -------------------------------------
def run_2DTFIM(numsteps, systemsize_x, systemsize_y, Bx=+1, num_units=50, numsamples=500,
               learningrate=5e-3, seed=666, print_assignment=False, Jz=None):
    # Seeding
    tf.compat.v1.reset_default_graph()
    random.seed(seed)  # `python` built-in pseudo-random generator
    np.random.seed(seed)  # numpy pseudo-random generator
    tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

    # Intitializing the RNN-----------
    units = [
        num_units]  # list containing the number of hidden units for each layer of the networks (We only support one layer for the moment)

    Nx = systemsize_x  # x dim
    Ny = systemsize_y  # y dim
    if Jz is None:
        Jz = nx.to_numpy_array(nx.gnp_random_graph(Nx*Ny, p, seed=seed))
    lr = np.float64(learningrate)

    input_dim = 2  # Dimension of the Hilbert space for each site (here = 2, up or down)
    numsamples_ = 20  # number of samples only for initialization
    wf = RNNwavefunction(Nx, Ny, units=units, cell=MDRNNcell, seed=seed)  # contains the graph with the RNNs

    # sampling=wf.sample(numsamples_,input_dim) #call this function once to create the dense layers

    # now initialize everything --------------------
    with wf.graph.as_default():
        samples_placeholder = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[numsamples_, Nx,
                                                                    Ny])  # the samples_placeholder are the samples of all of the spins
        global_step = tf.compat.v1.Variable(0, trainable=False)
        learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.compat.v1.float64, shape=[])
        learning_rate_withexpdecay = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step=global_step,
                                                                decay_steps=100, decay_rate=1.0,
                                                                staircase=True)  # For exponential decay of the learning rate (only works if decay_rate < 1.0)
        probs = wf.log_probability(samples_placeholder, input_dim)  # The probs are obtained by feeding the sample of spins.
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate_withexpdecay)  # Using AdamOptimizer
        init = tf.compat.v1.global_variables_initializer()
    # End Intitializing

    # Starting Session------------
    # Activating GPU
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.compat.v1.Session(graph=wf.graph, config=config)
    sess.run(init)
    # ---------------------------

    with wf.graph.as_default():
        variables_names = [v.name for v in tf.compat.v1.trainable_variables()]
        #     print(variables_names)
        sum = 0
        values = sess.run(variables_names)
        for k, v in zip(variables_names, values):
            v1 = tf.compat.v1.reshape(v, [-1])
            print(k, v1.shape)
            sum += v1.shape[0]
        print('The sum of params is {0}'.format(sum))

    meanEnergy = []
    varEnergy = []

    # Running the training -------------------
    path = os.getcwd()

    print('Training with numsamples = ', numsamples)
    print('\n')

    ending = 'units'
    # for u in units:
    #     ending += '_{0}'.format(u)
    # filename = '../Check_Points/2DTFIM/RNNwavefunction_2DVanillaRNN_' + str(Nx) + 'x' + str(Ny) + '_Bx' + str(
    #     Bx) + '_lradap' + str(lr) + '_samp' + str(numsamples) + ending + '.ckpt'
    # savename = '_2DTFIM'

    with tf.compat.v1.variable_scope(wf.scope, reuse=tf.compat.v1.AUTO_REUSE):
        with wf.graph.as_default():
            Eloc = tf.compat.v1.placeholder(dtype=tf.compat.v1.float64, shape=[numsamples])
            samp = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=[numsamples, Nx, Ny])
            log_probs_ = wf.log_probability(samp, inputdim=2)

            cost = tf.compat.v1.reduce_mean(tf.compat.v1.multiply(log_probs_, tf.compat.v1.stop_gradient(Eloc))) - tf.compat.v1.reduce_mean(
                tf.compat.v1.stop_gradient(Eloc)) * tf.compat.v1.reduce_mean(log_probs_)

            gradients, variables = zip(*optimizer.compute_gradients(cost))

            optstep = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            sess.run(tf.compat.v1.variables_initializer(optimizer.variables()))

            # saver = tf.compat.v1.train.Saver()  # define tf saver

    ##Loading previous trainings - Uncomment if you want to load the model----------
    # print("Loading the model")
    # with tf.compat.v1.variable_scope(wf.scope,reuse=tf.compat.v1.AUTO_REUSE):
    #     with wf.graph.as_default():
    #         saver.restore(sess,path+'/'+filename)
    #         meanEnergy=np.load('../Check_Points/2DTFIM/meanEnergy_2DVanillaRNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy').tolist()
    #         varEnergy=np.load('../Check_Points/2DTFIM/varEnergy_2DVanillaRNN_'+str(Nx)+'x'+ str(Ny) +'_Bx'+str(Bx)+'_lradap'+str(lr)+'_samp'+str(numsamples)+ending  + savename +'.npy').tolist()
    # -----------

    with tf.compat.v1.variable_scope(wf.scope, reuse=tf.compat.v1.AUTO_REUSE):
        with wf.graph.as_default():

            samples_ = wf.sample(numsamples=numsamples, inputdim=2)
            samples = np.ones((numsamples, Nx, Ny), dtype=np.int32)

            samples_placeholder = tf.compat.v1.placeholder(dtype=tf.compat.v1.int32, shape=(None, Nx, Ny))
            log_probs_tensor = wf.log_probability(samples_placeholder, inputdim=2)

            queue_samples = np.zeros((Nx * Ny + 1, numsamples, Nx, Ny),
                                     dtype=np.int32)  # Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
            log_probs = np.zeros((Nx * Ny + 1) * numsamples,
                                 dtype=np.float64)  # Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

            start_time = time.time()
            for it in range(len(meanEnergy), numsteps + 1):

                #                 print("sampling started")
                #                 start = time.time()
                samples = sess.run(samples_)
                #                 end = time.time()
                #                 print("sampling ended: "+ str(end - start))

                # Estimating local_energies
                local_energies = Ising2D_local_energies(Jz, Bx, Nx, Ny, samples, queue_samples, log_probs_tensor,
                                                        samples_placeholder, log_probs, sess)

                meanE = np.mean(local_energies)
                varE = np.var(local_energies)

                # adding elements to be saved
                meanEnergy.append(meanE)
                varEnergy.append(varE)

                # if it % 10 == 0:
                #     print('mean(E): {0}, var(E): {1}, #samples {2}, #Step {3} \n\n'.format(meanE, varE, numsamples, it))

                # Comment if you dont want to save or if saving is not working
                # if it % 500 == 0:  # 500 can be changed to suite your chosen number of iterations and to avoid slow down by saving the model too often
                #     # Saving the model
                #     saver.save(sess, path + '/' + filename)
                #
                # # Comment if you dont want to save or if saving is not working
                # if it % 10 == 0:
                #     # Saving the performances
                #     np.save('../Check_Points/2DTFIM/meanEnergy_2DVanillaRNN_' + str(Nx) + 'x' + str(Ny) + '_Bx' + str(
                #         Bx) + '_lradap' + str(lr) + '_samp' + str(numsamples) + ending + savename + '.npy', meanEnergy)
                #     np.save('../Check_Points/2DTFIM/varEnergy_2DVanillaRNN_' + str(Nx) + 'x' + str(Ny) + '_Bx' + str(
                #         Bx) + '_lradap' + str(lr) + '_samp' + str(numsamples) + ending + savename + '.npy', varEnergy)

                # lr_adaptation
                lr_adapted = lr * (1 + it / 5000) ** (-1)
                # Optimize
                sess.run(optstep, feed_dict={Eloc: local_energies, samp: samples, learningrate_placeholder: lr_adapted})
            end_time = time.time()


    MIS_size = 0
    fin_samp = sess.run(samples_)
    assignment = np.zeros(fin_samp.shape[1])
    for i in range(fin_samp.shape[1]):
        assignment[i] = np.sum(fin_samp[:,i,0])/fin_samp.shape[0]

    G = nx.from_numpy_matrix(Jz)
    pos = nx.circular_layout(G)
    color = []
    for i in range(Nx*Ny):
        if round(assignment[i]) == 1:
            MIS_size += 1
            color.append('red')
        else:
            color.append('blue')

    if print_assignment:
        print(assignment)
        nx.draw(G, pos=pos, node_color=color)
        plt.title("Node Assignment")
        plt.show()

    return meanEnergy, varEnergy, end_time-start_time, assignment, MIS_size
