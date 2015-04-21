import math
import random
import copy


def log_func(x):
    """
    The transfer function of neurons, g(x)

    :param x:
    :return:
    """
    return 1.0 / (1.0 + math.exp(-x))


def log_func_derivative(x):
    """
    The derivative of the transfer function, g'(x)

    :param x:
    :return:
    """
    return math.exp(-x) / (pow(math.exp(-x) + 1, 2))


def random_float(low, high):
    """
    Returns a random float between the two limits

    :param low: Lower limit
    :param high: Upper limit
    :return: A random float
    """
    return random.random() * (high - low) + low


def make_matrix(i, j):
    """
    Initializes a matrix of all zeros

    :param i: First dimension
    :param j: Second dimension
    :return: A matrix of all zeroes
    """
    m = []

    for u in range(i):
        m.append([0] * j)

    return m


class NN:
    """
    Class holding a neural Network

    :
    """

    def __init__(self, num_inputs, num_hidden, learning_rate=0.001):
        """
        Initializes a neural network. Assumes a single output node.

        :param num_inputs: Number of input nodes.
        :param num_hidden: Number of hidden nodes.
        :param learning_rate: The learning rate.
        :return: A neural network that follows the specified inputs.
        """

        self.num_inputs = num_inputs + 1  # Add a bias node (constant input of 1). Used to shift the transfer function.
        self.num_hidden = num_hidden

        # Current activation levels for nodes (in other words, the nodes' output value)
        self.out_i_b = [1.0] * self.num_inputs  # Corresponds to out_i_b
        self.out_h_b = [1.0] * self.num_hidden  # Corresponds to out_h_b
        self.o_b = 1.0  # Corresponds to o_b
        self.learning_rate = learning_rate

        # Create weights
        # A matrix with all weights from input layer to hidden layer
        self.w_i_h = make_matrix(self.num_inputs, self.num_hidden)

        for i in range(self.num_inputs):  # Set the matrix to random values
            for j in range(self.num_hidden):
                self.w_i_h[i][j] = random_float(-0.5, 0.5)

        # A list with all weights from hidden layer to the single output neuron.
        self.w_h_o = [0] * self.num_hidden  # Corresponds to w_h_o

        for j in range(self.num_hidden):  # Set the list to random values
            self.w_h_o[j] = random_float(-0.5, 0.5)

        # Data for the back-propagation step in RankNets.
        # For storing the previous activation levels (output levels) of all neurons
        self.out_i_a = []  # Corresponds to out_i_a
        self.out_h_a = []  # Corresponds to out_h_a
        self.o_a = 0  # Corresponds to o_a

        # For storing the previous delta in the output and hidden layer
        self.delta_o_a = 0  # Corresponds to delta_o_a
        self.delta_h_a = [0] * self.num_hidden  # Corresponds to delta_h_a

        # For storing the current delta in the same layers
        self.delta_o_b = 0  # Corresponds to delta_o_b
        self.delta_h_b = [0] * self.num_hidden  # Corresponds to delta_h_b (list for all nodes)

    def propagate(self, inputs):
        if len(inputs) != self.num_inputs - 1:
            raise ValueError('Wrong number of inputs')

        self.out_i_a = copy.deepcopy(self.out_i_b)  # Save old input activations

        for i in range(self.num_inputs - 1):  # Replace by new input activations
            self.out_i_b[i] = inputs[i]

        self.out_i_b[-1] = 1  # Set bias node to -1.

        self.out_h_a = copy.deepcopy(self.out_h_b)  # Save hidden input activations

        for j in range(self.num_hidden):  # Calculate new activations for each hidden node
            hidden_node_sum = 0.0

            for i in range(self.num_inputs):
                hidden_node_sum += self.out_i_b[i] * self.w_i_h[i][j]

            self.out_h_b[j] = log_func(hidden_node_sum)

        self.o_a = self.o_b  # Save old output activation

        output_node_sum = 0.0

        for j in range(self.num_hidden):  # Calculate output activation
            output_node_sum += self.out_h_b[j] * self.w_h_o[j]

        self.o_b = log_func(output_node_sum)

        return self.o_b

    def compute_output_delta(self):
        """
        Computes the output delta given on P3 of the exercise. Assumes that two results have been fed into
        the network, and that the first pair had a higher rating.
        """

        o_a = self.o_a  # Output level of the highest rated pair
        o_b = self.o_b  # Output level of the lowest rated pair

        p_ab = 1 / (1 + math.exp(o_b - o_a))

        delta_o_a = log_func_derivative(o_a) * (1 - p_ab)
        delta_o_b = log_func_derivative(o_b) * (1 - p_ab)

        self.delta_o_a = delta_o_a
        self.delta_o_b = delta_o_b

    def compute_hidden_delta(self):
        """
        Update the deltas for the hidden nodes.
        """
        delta_o_a = self.delta_o_a
        delta_o_b = self.delta_o_b

        for i in range(self.num_hidden):  # update h_a
            self.delta_h_a[i] = log_func_derivative(self.out_h_a[i]) * \
                                self.w_h_o[i] * (delta_o_a - delta_o_b)

        for i in range(self.num_hidden):  # update h_b
            self.delta_h_b[i] = log_func_derivative(self.out_h_b[i]) * \
                                self.w_h_o[i] * (delta_o_a - delta_o_b)

    def update_weights(self):
        """
        Updates the weights for the network.
        """

        for i in range(self.num_inputs):
            for j in range(self.num_hidden):
                self.w_i_h[i][j] += self.learning_rate * (
                    self.delta_h_a[j] * self.out_i_a[i] - self.delta_h_b[j] * self.out_i_b[i])

        for h in range(self.num_hidden):
            self.w_h_o[h] += self.learning_rate * (self.delta_o_a * self.out_h_a[h] - self.delta_o_b * self.out_h_b[h])
        # TODO: Update the weights of the network using the deltas (see exercise text)

        pass

    def back_propagate(self):
        self.compute_output_delta()
        self.compute_hidden_delta()
        self.update_weights()

    def weights(self):
        """Prints the network weights.

        :return:
        """
        print('Input weights:')

        for i in range(self.num_inputs):
            print(self.w_i_h[i])

        print()
        print('Output weights:')
        print(self.w_h_o)

    def train(self, patterns, iterations=1):

        error_rates = list()

        for i in range(iterations):
            for a, b in patterns:
                self.propagate(a)
                self.propagate(b)
                self.back_propagate()

            error_rates.append(self.count_misordered_pairs(patterns))

        return error_rates

    def count_misordered_pairs(self, patterns):
        errors = 0

        for a, b in patterns:
            result_a = self.propagate(a)
            result_b = self.propagate(b)
            if result_a < result_b:
                errors += 1

        return errors / len(patterns)