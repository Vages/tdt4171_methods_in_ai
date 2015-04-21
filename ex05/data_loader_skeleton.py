__author__ = 'kaiolae'
import backprop_skeleton as bp
import time
import matplotlib.pyplot as plt


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


def load_data(file_path):
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

        q_inst = QueryInstance(qid, instance_rating, instance_features)  # Creating a new query instance, inserting in the dict.
        if qid in query_dict:
            query_dict[qid].append(q_inst)
        else:
            query_dict[qid] = [q_inst]

    return query_dict


def generate_sorted_data_set_tuples(query_dict):
    """Finds all possible QueryInstance pairs for all queries in a query dictionary. The pairs must be for the same
    query ID, and the first must have a higher rating than the second. It strips the pairs of anything but their
    features.

    :param query_dict: A dictionary mapping query_id to a  list of relevant QueryInstances
    :return: All possible feature pairs given the query
    """
    results = list()

    for query_id in query_dict:
        # This iterates through every query ID in our training set
        query_instances = query_dict[query_id]  # All data instances (query, features, rating) for query query_id

        # Split the examples by rating
        instances_split_by_rating = dict()
        for instance in query_instances:
            key = instance.rating
            if key in instances_split_by_rating:
                instances_split_by_rating[key].append(instance)
            else:
                instances_split_by_rating[key] = [instance]

        rating_values = sorted(instances_split_by_rating.keys(), reverse=True)  # Sort rating values in descending order

        for i in range(len(rating_values)-1):
            for j in range(i+1, len(rating_values)):
                for higher_ranked_item in instances_split_by_rating[rating_values[i]]:
                    for lower_ranked_item in instances_split_by_rating[rating_values[j]]:
                        results.append((higher_ranked_item.features, lower_ranked_item.features))

    return results

def average_lists(list_of_lists, invert=False):
    """Finds the average value of the elements at each list index in a list.

    :param args: A variable number of lists. All must have equal length
    :return: A list of the averages
    """

    print(list_of_lists)
    averages = list()
    no_of_elements = len(list_of_lists)

    for i in range(len(list_of_lists[0])):
        results_sum = 0
        for item in list_of_lists:
            print(item[i])
            results_sum += item[i]

        if invert:
            results_sum = no_of_elements-results_sum
        averages.append(results_sum/no_of_elements)

    return averages


def plot_errors(*args):
    x_axis = [i for i in range(len(args[0]))]
    plt.plot(x_axis, args[0], "b", x_axis, args[1], "r--")
    plt.ylim([0, 1])

    plt.title('Error ratios by epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio')

    plt.show()

def run_rank(training_set, test_set, learning_rate=0.001, epochs=25):
    """

    :param training_set: File path to training set.
    :param test_set: File path to test set.
    :param learning_rate: Learning rate of the neural network.
    :param epochs:
    :return:
    """

    # Make data sets for training and testing. Sort results for each query ID by rating. (Descending rating.)
    data_set_training = load_data(training_set)
    data_set_testing = load_data(test_set)

    nn = bp.NN(46, 10, learning_rate)  # Create an ANN instance

    # The lists below should hold training patterns in this format:
    # [(data1Features, data2Features), (data1Features, data3Features), ... , (dataNFeatures, dataMFeatures)]
    # The training set needs to have pairs ordered so the first item of the pair has a higher rating.

    training_patterns = generate_sorted_data_set_tuples(data_set_training)
    testing_patterns = generate_sorted_data_set_tuples(data_set_testing)

    # Check ANN performance before training
    print("\nTraining neural network")
    print("\n\tLearning rate:", learning_rate)
    print("\tIterations:", epochs)
    print("\n\tTraining set size:", len(training_patterns))
    print("\tTest set size:", len(testing_patterns))
    a = time.time()

    performance_on_testing_patterns_before_training = nn.count_misordered_pairs(testing_patterns)
    print("\nPerformance on test set before training:", performance_on_testing_patterns_before_training)
    training_errors = [nn.count_misordered_pairs(training_patterns)]
    testing_errors = [performance_on_testing_patterns_before_training]
    for i in range(epochs):
        # Running 25 epochs, measuring testing performance after each round of training.
        # Training
        training_errors.append(nn.train(training_patterns, iterations=1)[0])
        # Check ANN performance after training.
        testing_errors.append(nn.count_misordered_pairs(testing_patterns))
        j = i
        print('\nTraining error epoch %d:' % (j+1), training_errors[j])
        print('Testing error epoch %d:' % (j+1), testing_errors[j])

    b = time.time()

    print('\nFinished training and testing in %.2f minutes.' % ((b-a)/60))

    return training_errors, testing_errors


def average_run_rank(training_set, test_set, learning_rate=0.001, epochs=25, runs=5):
    """
    Does a few runs of the run_rank algorithm and averages the results

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
        print("\nRun %d of %d" % (i+1, runs))
        x, y = run_rank(training_set, test_set, learning_rate, epochs)
        training_error_rates.append(x)
        testing_error_rates.append(y)

    average_training_error_rates = average_lists(training_error_rates, invert=True)
    average_testing_error_rates = average_lists(testing_error_rates, invert=True)

    plot_errors(average_training_error_rates, average_testing_error_rates)

if __name__ == '__main__':
    print("Running")

    #plot_errors([0.5, 0.75, 0.3]*5, [0.2, 0.8, 0.6]*5)
    average_run_rank("data_sets/train.txt", "data_sets/test.txt", learning_rate=0.001, epochs=10, runs=1)
