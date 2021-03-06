import tensorflow as tf
import numpy as np
import random


class RNNwavefunction(object):
    def __init__(self, systemsize_x, systemsize_y, cell=None, units=[10], scope='RNNwavefunction', seed=111):
        """
            systemsize_x:  int
                         number of sites for x-axis
            systemsize_y:  int
                         number of sites for y-axis
            cell:        a tensorflow RNN cell
            units:       list of int
                         number of units per RNN layer
            scope:       str
                         the name of the name-space scope
            seed:        pseudo-random number generator
        """
        self.graph = tf.compat.v1.Graph()
        self.scope = scope  # Label of the RNN wavefunction
        self.Nx = systemsize_x  # size of x direction in the 2d model
        self.Ny = systemsize_y

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        # Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
                tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator
                self.rnn = cell(num_units=units[0], num_in=2, name="rnn_" + str(0), dtype=tf.compat.v1.float64)
                self.dense = tf.compat.v1.layers.Dense(2, activation=tf.compat.v1.nn.softmax, name='wf_dense', dtype=tf.compat.v1.float64)

    def sample(self, numsamples, inputdim):
        """
            generate samples from a probability distribution parametrized by a recurrent network
            ------------------------------------------------------------------------
            Parameters:

            numsamples:      int



                             samples to be produced
            inputdim:        int
                             hilbert space dimension of one spin

            ------------------------------------------------------------------------
            Returns:

            samples:         tf.compat.v1.Tensor of shape (numsamples,systemsize_x, systemsize_y)
                             the samples in integer encoding
        """

        with self.graph.as_default():  # Call the default graph, used if not willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):

                # Initial input to feed to the 2drnn

                self.inputdim = inputdim
                self.outputdim = self.inputdim
                self.numsamples = numsamples

                "changed this"
                samples = [[[] for ny in range(self.Ny)] for nx in range(self.Nx)]
                rnn_states = {}
                inputs = {}

                for ny in range(self.Ny):  # Loop over the boundary
                    if ny % 2 == 0:
                        nx = -1
                        # print(nx,ny)
                        rnn_states[str(nx) + str(ny)] = self.rnn.zero_state(self.numsamples, dtype=tf.compat.v1.float64)
                        inputs[str(nx) + str(ny)] = tf.compat.v1.zeros((self.numsamples, inputdim), dtype=tf.compat.v1.float64)

                    if ny % 2 == 1:
                        nx = self.Nx
                        # print(nx,ny)
                        rnn_states[str(nx) + str(ny)] = self.rnn.zero_state(self.numsamples, dtype=tf.compat.v1.float64)
                        inputs[str(nx) + str(ny)] = tf.compat.v1.zeros((self.numsamples, inputdim), dtype=tf.compat.v1.float64)

                for nx in range(self.Nx):  # Loop over the boundary
                    ny = -1
                    rnn_states[str(nx) + str(ny)] = self.rnn.zero_state(self.numsamples, dtype=tf.compat.v1.float64)
                    inputs[str(nx) + str(ny)] = tf.compat.v1.zeros((self.numsamples, inputdim), dtype=tf.compat.v1.float64)

                    # Begin sampling
                for ny in range(self.Ny):

                    if ny % 2 == 0:

                        for nx in range(self.Nx):  # left to right

                            rnn_output, rnn_states[str(nx) + str(ny)] = self.rnn(
                                (inputs[str(nx - 1) + str(ny)], inputs[str(nx) + str(ny - 1)]),
                                (rnn_states[str(nx - 1) + str(ny)], rnn_states[str(nx) + str(ny - 1)]))

                            output = self.dense(rnn_output)
                            sample_temp = tf.compat.v1.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output), num_samples=1), [-1, ])
                            samples[nx][ny] = sample_temp
                            inputs[str(nx) + str(ny)] = tf.compat.v1.one_hot(sample_temp, depth=self.outputdim, dtype=tf.compat.v1.float64)

                    if ny % 2 == 1:

                        for nx in range(self.Nx - 1, -1, -1):  # right to left

                            rnn_output, rnn_states[str(nx) + str(ny)] = self.rnn(
                                (inputs[str(nx + 1) + str(ny)], inputs[str(nx) + str(ny - 1)]),
                                (rnn_states[str(nx + 1) + str(ny)], rnn_states[str(nx) + str(ny - 1)]))

                            output = self.dense(rnn_output)
                            sample_temp = tf.compat.v1.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output), num_samples=1), [-1, ])
                            samples[nx][ny] = sample_temp
                            inputs[str(nx) + str(ny)] = tf.compat.v1.one_hot(sample_temp, depth=self.outputdim, dtype=tf.compat.v1.float64)

        self.samples = tf.compat.v1.transpose(tf.compat.v1.stack(values=samples, axis=0), perm=[2, 0, 1])

        return self.samples

    def log_probability(self, samples, inputdim):
        """
            calculate the log-probabilities of ```samples``
            ------------------------------------------------------------------------
            Parameters:

            samples:         tf.compat.v1.Tensor
                             a tf.compat.v1.placeholder of shape (number of samples,systemsize_x,system_size_y)
                             containing the input samples in integer encoding
            inputdim:        int
                             dimension of the input space

            ------------------------------------------------------------------------
            Returns:
            log-probs        tf.compat.v1.Tensor of shape (number of samples,)
                             the log-probability of each sample
            """
        with self.graph.as_default():

            self.inputdim = inputdim
            self.outputdim = self.inputdim
            self.numsamples = tf.compat.v1.shape(samples)[0]
            self.outputdim = self.inputdim

            samples_ = tf.compat.v1.transpose(samples, perm=[1, 2, 0])
            rnn_states = {}
            inputs = {}

            for ny in range(self.Ny):  # Loop over the boundary
                if ny % 2 == 0:
                    nx = -1
                    rnn_states[str(nx) + str(ny)] = self.rnn.zero_state(self.numsamples, dtype=tf.compat.v1.float64)
                    inputs[str(nx) + str(ny)] = tf.compat.v1.zeros((self.numsamples, inputdim), dtype=tf.compat.v1.float64)

                if ny % 2 == 1:
                    nx = self.Nx
                    rnn_states[str(nx) + str(ny)] = self.rnn.zero_state(self.numsamples, dtype=tf.compat.v1.float64)
                    inputs[str(nx) + str(ny)] = tf.compat.v1.zeros((self.numsamples, inputdim), dtype=tf.compat.v1.float64)

            for nx in range(self.Nx):  # Loop over the boundary
                ny = -1
                rnn_states[str(nx) + str(ny)] = self.rnn.zero_state(self.numsamples, dtype=tf.compat.v1.float64)
                inputs[str(nx) + str(ny)] = tf.compat.v1.zeros((self.numsamples, inputdim), dtype=tf.compat.v1.float64)

            with tf.compat.v1.variable_scope(self.scope, reuse=tf.compat.v1.AUTO_REUSE):
                " changed this line"
                probs = [[[] for ny in range(self.Ny)] for nx in range(self.Nx)]

                # Begin estimation of log probs
                for ny in range(self.Ny):

                    if ny % 2 == 0:

                        for nx in range(self.Nx):  # left to right

                            rnn_output, rnn_states[str(nx) + str(ny)] = self.rnn(
                                (inputs[str(nx - 1) + str(ny)], inputs[str(nx) + str(ny - 1)]),
                                (rnn_states[str(nx - 1) + str(ny)], rnn_states[str(nx) + str(ny - 1)]))

                            output = self.dense(rnn_output)
                            sample_temp = tf.compat.v1.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output), num_samples=1), [-1, ])
                            probs[nx][ny] = output
                            inputs[str(nx) + str(ny)] = tf.compat.v1.one_hot(samples_[nx, ny], depth=self.outputdim,
                                                                   dtype=tf.compat.v1.float64)

                    if ny % 2 == 1:

                        for nx in range(self.Nx - 1, -1, -1):  # right to left

                            rnn_output, rnn_states[str(nx) + str(ny)] = self.rnn(
                                (inputs[str(nx + 1) + str(ny)], inputs[str(nx) + str(ny - 1)]),
                                (rnn_states[str(nx + 1) + str(ny)], rnn_states[str(nx) + str(ny - 1)]))

                            output = self.dense(rnn_output)
                            # sample_temp = tf.compat.v1.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output), num_samples=1), [-1, ])
                            probs[nx][ny] = output
                            inputs[str(nx) + str(ny)] = tf.compat.v1.one_hot(samples_[nx, ny], depth=self.outputdim,
                                                                   dtype=tf.compat.v1.float64)

            probs = tf.compat.v1.transpose(tf.compat.v1.stack(values=probs, axis=0), perm=[2, 0, 1, 3])
            one_hot_samples = tf.compat.v1.one_hot(samples, depth=self.inputdim, dtype=tf.compat.v1.float64)

            self.log_probs = tf.compat.v1.reduce_sum(
                tf.compat.v1.reduce_sum(tf.compat.v1.log(tf.compat.v1.reduce_sum(tf.compat.v1.multiply(probs, one_hot_samples), axis=3)), axis=2), axis=1)

            return self.log_probs
