import math
import random
import copy


def log_func(x):
    """
    The transfer function of neurons, g(x)

    :param x:
    :return:
    """
    return 1.0/(1.0+math.exp(-x))


def log_func_derivative(x):
    """
    The derivative of the transfer function, g'(x)

    :param x:
    :return:
    """
    return math.exp(-x)/(pow(math.exp(-x)+1, 2))


def random_float(low, high):
    """
    Returns a random float between the two limits

    :param low: Lower limit
    :param high: Upper limit
    :return: A random float
    """
    return random.random()*(high-low) + low


def make_matrix(i, j):
    """
    Initializes a matrix of all zeros

    :param i: First dimension
    :param j: Second dimension
    :return: A matrix of all zeroes
    """
    m = []

    for u in range(i):
        m.append([0]*j)

    return m


class NN:
    """
    Class holding a neural Network
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
        self.input_activation = [1.0] * self.num_inputs
        self.hidden_activations = [1.0] * self.num_hidden
        self.output_activation = 1.0  # Assuming a single output.
        self.learning_rate = learning_rate

        # create weights
        # A matrix with all weights from input layer to hidden layer
        self.weights_input = make_matrix(self.num_inputs, self.num_hidden)

        # A list with all weights from hidden layer to the single output neuron.
        self.weights_output = [0 for i in range(self.num_hidden)]  # Assuming single output

        for i in range(self.num_inputs):  # Set them to random values
            for j in range(self.num_hidden):
                self.weights_input[i][j] = random_float(-0.5, 0.5)

        for j in range(self.num_hidden):
            self.weights_output[j] = random_float(-0.5, 0.5)

        # Data for the back-propagation step in RankNets.
        # For storing the previous activation levels (output levels) of all neurons
        self.prev_input_activations = []
        self.prev_hidden_activations = []
        self.prev_output_activation = 0

        # For storing the previous delta in the output and hidden layer
        self.prev_delta_output = 0
        self.prev_delta_hidden = [0 for i in range(self.num_hidden)]

        # For storing the current delta in the same layers
        self.delta_output = 0
        self.delta_hidden = [0 for i in range(self.num_hidden)]

    def propagate(self, inputs):
        if len(inputs) != self.num_inputs-1:
            raise ValueError('Wrong number of inputs')

        # input activations
        self.prev_input_activations = copy.deepcopy(self.input_activation)

        for i in range(self.num_inputs-1):
            self.input_activation[i] = inputs[i]

        self.input_activation[-1] = 1  # Set bias node to -1.

        # hidden activations
        self.prev_hidden_activations = copy.deepcopy(self.hidden_activations)

        for j in range(self.num_hidden):
            sum = 0.0

            for i in range(self.num_inputs):
                # print self.ai[i] ," * " , self.wi[i][j]
                sum += self.input_activation[i] * self.weights_input[i][j]

            self.hidden_activations[j] = log_func(sum)

        # output activations
        self.prev_output_activation = self.output_activation

        sum = 0.0

        for j in range(self.num_hidden):
            sum += self.hidden_activations[j] * self.weights_output[j]

        self.output_activation = log_func(sum)

        return self.output_activation

    def compute_output_delta(self):
        pass
        # TODO: Implement the delta function for the output layer (see exercise text)

    def compute_hidden_delta(self):
        pass
        # TODO: Implement the delta function for the hidden layer (see exercise text)

    def update_weights(self):
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
            print(self.weights_input[i])

        print()
        print('Output weights:')
        print(self.weights_output)

    def train(self, patterns, iterations=1):
        # TODO: Train the network on all patterns for a number of iterations.
        # To measure performance each iteration: Run for 1 iteration, then count misordered pairs.
        # TODO: Training is done  like this (details in exercise text):
        # -Propagate A
        # -Propagate B
        # -Back-propagate
        pass

    def count_misordered_pairs(self, patterns):
        # TODO: Let the network classify all pairs of patterns. The highest output determines the winner.
        # for each pair, do
        # Propagate A
        # Propagate B
        # if A>B: A wins. If B>A: B wins
        # if rating(winner) > rating(loser): numRight++
        # else: numMisses++
        # end of for
        # TODO: Calculate the ratio of correct answers:
        # errorRate = numMisses/(numRight+numMisses)
        pass