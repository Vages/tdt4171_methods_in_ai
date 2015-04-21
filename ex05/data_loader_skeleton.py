__author__ = 'kaiolae'
import backprop_skeleton as bp


class DataInstance:
    """
    Class for holding your data - one object for each line in the data set
    """

    def __init__(self, qid, rating, features):
        self.qid = qid  # ID of the query
        self.rating = rating  # Rating of this site for this query
        self.features = features  # The features of this query-site pair.

    def __str__(self):
        return "Data instance - qid: " + str(self.qid) + ". rating: " + str(self.rating) \
               + ". features: " + str(self.features)


def load_data(file_path):
    """Loads data from a file and returns it as a dict mapping each query ID to a list of relevant documents.

    :param file_path: A file with the data.
    :return: A dict mapping each query ID to the relevant documents, like this: data_set[queryID] = [dataInstance1, ...]
    """

    data = open(file_path)
    data_set = {}
    for line in data:  # Extracting all the useful info from the line of data
        line_data = line.split()
        rating = int(line_data[0])
        qid = int(line_data[1].split(':')[1])
        features = []
        for elem in line_data[2:]:
            if '#docid' in elem:  # We reached a comment. Line done.
                break
            features.append(float(elem.split(':')[1]))

        di = DataInstance(qid, rating, features)  # Creating a new data instance, inserting in the dict.
        if qid in data_set:
            data_set[qid].append(di)
        else:
            data_set[qid] = [di]

    return data_set


# Eirik: Perhaps simply remove this class and work directly on the data
class DataHolder:
    """
    A class that holds all the data in one of our sets (the training set or the test set)
    """
    def __init__(self, data_set):
        self.data_set = load_data(data_set)


def generate_sorted_data_set_tuples(data_set):
    """

    :param data_set: A dictionary mapping (query_id, list(DataInstance))
    :return: All tuples of rankings possible from each query
    """
    results = []

    for qid in data_set:
        # This iterates through every query ID in our training set
        data_instance = data_set[qid]  # All data instances (query, features, rating) for query qid

        # Split the examples by rating
        data_split_by_rating = {}
        for item in data_instance:
            key = item.rating
            if key in data_split_by_rating:
                data_split_by_rating[key].append(item)
            else:
                data_split_by_rating[key] = [item]

        rating_values = sorted(data_split_by_rating.keys(), reverse=True)  # Sort possible values in descending order

        # For each item of each key in each set, generate the
        for i in range(len(rating_values)-1):
            for j in range(i+1, len(rating_values)):
                for higher_ranked_item in data_split_by_rating[rating_values[i]]:
                    for lower_ranked_item in data_split_by_rating[rating_values[j]]:
                        a = str(higher_ranked_item)
                        b = str(lower_ranked_item)
                        results.append((higher_ranked_item, lower_ranked_item))

    return results

def run_rank(training_set, test_set, learning_rate=0.001, iterations=25):
    """

    :param training_set: File path to training set.
    :param test_set: File path to test set.
    :param learning_rate: Learning rate of the neural network.
    :param iterations:
    :return:
    """

    # TODO: Insert the code for training and testing your ranker here.

    # Make data sets for training and testing. Sort results for each query ID by rating. (Descending rating.)
    data_set_training = load_data(training_set)
    #data_set_testing = load_data(test_set)

    nn = bp.NN(46, 10, learning_rate)  # Create an ANN instance
    # TODO: Feel free to experiment with the learning rate (the third parameter).

    # The lists below should hold training patterns in this format:
    # [(data1Features, data2Features), (data1Features, data3Features), ... , (dataNFeatures, dataMFeatures)]
    # The training set needs to have pairs ordered so the first item of the pair has a higher rating.

    training_patterns = generate_sorted_data_set_tuples(data_set_training)
    #test_patterns = generate_sorted_data_set_tuples(data_set_testing)


    # Check ANN performance before training
    nn.count_misordered_pairs(test_patterns)
    for i in range(iterations):
        # Running 25 iterations, measuring testing performance after each round of training.
        # Training
        nn.train(training_patterns, iterations=1)
        # Check ANN performance after training.
        nn.count_misordered_pairs(test_patterns)

    # TODO: Store the data returned by count_misordered_pairs and plot it,
    # showing how training and testing errors develop.


if __name__ == '__main__':
    pass
    run_rank("data_sets/train.txt", "data_sets/test.txt")
