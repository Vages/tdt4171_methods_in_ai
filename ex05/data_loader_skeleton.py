__author__ = 'kaiolae'
import time

import matplotlib.pyplot as plt

import backprop_skeleton as bp


class QueryInstance:
    """
    Holds a query, its rating, and its features
    """

    def __init__(self, qid, rating, features):
        self.qid = qid
        self.rating = rating
        self.features = features

    def __str__(self):
        return "Data instance - qid: " + str(self.qid) + ". rating: " + str(self.rating) \
               + ". features: " + str(self.features)


def load_query_dict_from_file(file_path):
    """Loads queries from a file and returns a dict mapping each query ID to a list of relevant QueryInstances.

    :param file_path: A file with the data.
    :return: A dict mapping query IDs to relevant QueryInstances: query_dict[queryID] = [query_instance1, ...]
    """

    queries_file = open(file_path)
    query_dict = {}
    for line in queries_file:
        # Fields of query given by position
        instance_data = line.split()
        instance_rating = int(instance_data[0])
        qid = int(instance_data[1].split(':')[1])
        instance_features = []
        for elem in instance_data[2:]:
            if '#docid' in elem:  # Reached a comment. Line done.
                break
            instance_features.append(float(elem.split(':')[1]))

        q_inst = QueryInstance(qid, instance_rating,
                               instance_features)  # Creating a new query instance, inserting in the dict.
        if qid in query_dict:
            query_dict[qid].append(q_inst)
        else:
            query_dict[qid] = [q_inst]

    return query_dict


def generate_sorted_feature_pairs(query_dict):
    """Finds all possible QueryInstance pairs for all queries in a query dictionary. It strips the pairs of anything
    but their features. The pairs will be for the same query ID, and the first will have a higher rating than the
    second.

    :param query_dict: A dictionary mapping query_id to a list of relevant QueryInstances
    :return: All possible feature pairs given the query
    """
    results = list()

    for query_id in query_dict:
        # This iterates through every query ID in our training set
        query_instances = query_dict[query_id]  # All query instances for the query_id

        # Split the examples by rating
        instances_split_by_rating = dict()
        for instance in query_instances:
            key = instance.rating
            if key in instances_split_by_rating:
                instances_split_by_rating[key].append(instance)
            else:
                instances_split_by_rating[key] = [instance]

        # Find all rating values and sort them in descending orders
        rating_values = sorted(instances_split_by_rating.keys(), reverse=True)

        # Generate every possible pair
        for i in range(len(rating_values) - 1):
            for j in range(i + 1, len(rating_values)):
                for higher_ranked_item in instances_split_by_rating[rating_values[i]]:
                    for lower_ranked_item in instances_split_by_rating[rating_values[j]]:
                        results.append((higher_ranked_item.features, lower_ranked_item.features))

    return results


def average_lists(list_of_lists, invert=False):
    """Finds the average value of the elements at a given position (given several lists of equal length).

    :param args: A variable number of lists. All must have equal length.
    :param invert: Whether the values should be inverted, i.e. subtracted from 1.
    :return: A list with the average values.
    """

    averages = list()  # Will contain the averages
    no_of_lists = len(list_of_lists)  # Length of lists

    for i in range(len(list_of_lists[0])):
        results_sum = 0
        for item in list_of_lists:
            results_sum += item[i]

        if invert:
            results_sum = no_of_lists - results_sum

        averages.append(results_sum / no_of_lists)

    return averages


def plot_errors(*args):
    """Shows two lists of equal length in a plot.

    :param args: The two lists
    :return:
    """

    x_axis = [i for i in range(len(args[0]))]  # Generate x-coordinates
    plt.plot(x_axis, args[0], "b", x_axis, args[1], "r--")
    plt.ylim([0, 1])  # Set y-axis limits

    plt.title('Error ratios by epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')

    plt.show()


def run_ranknet(training_set, test_set, learning_rate=0.001, epochs=25):
    """Runs the the RankNet algorithm for the given number of epochs and returns a tuple of training set error rates
    and test set error rates, listed for each epoch.

    :param training_set: File path to training set
    :param test_set: File path to test set
    :param learning_rate: Learning rate of the neural network
    :param epochs: Number of epochs that the set will be run for
    :return: Training set error rates, test set error rates
    """

    query_dict_training = load_query_dict_from_file(training_set)
    query_dict_testing = load_query_dict_from_file(test_set)

    nn = bp.NN(46, 10, learning_rate)  # Create an artificial neural network

    training_pairs = generate_sorted_feature_pairs(query_dict_training)
    testing_pairs = generate_sorted_feature_pairs(query_dict_testing)

    # Check ANN performance before training
    print("\nTraining neural network")
    print("\n\tLearning rate:", learning_rate)
    print("\tIterations:", epochs)
    print("\n\tTraining set size:", len(training_pairs))
    print("\tTest set size:", len(testing_pairs))

    a = time.time()  # For measuring time taken

    # Check performance before training
    performance_on_training_pairs_before_training = nn.count_misordered_pairs(training_pairs)
    performance_on_testing_pairs_before_training = nn.count_misordered_pairs(testing_pairs)

    print("\nPerformance on test set before training:", performance_on_testing_pairs_before_training)

    training_errors = [performance_on_training_pairs_before_training]
    testing_errors = [performance_on_testing_pairs_before_training]

    for i in range(epochs):
        training_errors.append(nn.train(training_pairs, iterations=1)[0])
        testing_errors.append(nn.count_misordered_pairs(testing_pairs))  # Check ANN performance after training.

        print('\nTraining error epoch %d:' % (i + 1), training_errors[i])
        print('Testing error epoch %d:' % (i + 1), testing_errors[i])

    b = time.time()

    print('\nFinished training and testing in %.2f minutes.' % ((b - a) / 60))

    return training_errors, testing_errors


def average_run_ranknet(training_set, test_set, learning_rate=0.001, epochs=25, runs=5):
    """
    Does a few runs of the run_ranknet algorithm with the same parameters and averages the results.

    :param training_set:
    :param test_set:
    :param learning_rate:
    :param epochs:
    :param runs:
    :return:
    """

    training_error_rates = list()
    testing_error_rates = list()

    for i in range(runs):
        print("\nRun %d of %d" % (i + 1, runs))
        x, y = run_ranknet(training_set, test_set, learning_rate, epochs)
        training_error_rates.append(x)
        testing_error_rates.append(y)

    average_training_error_rates = average_lists(training_error_rates, invert=True)
    average_testing_error_rates = average_lists(testing_error_rates, invert=True)

    plot_errors(average_training_error_rates, average_testing_error_rates)


if __name__ == '__main__':
    print("Running")

    average_run_ranknet("data_sets/train.txt", "data_sets/test.txt", learning_rate=0.001, epochs=25, runs=5)
