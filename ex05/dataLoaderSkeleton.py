__author__ = 'kaiolae'
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
    """Loads data from a file

    :param file_path: A file with the data.
    :return: A dict mapping each query ID to the relevant documents,
    like this: data_set[queryID] = [dataInstance1, dataInstance2, ...]
    """
    data = open(file_path)
    data_set = {}
    for line in data:
        # Extracting all the useful info from the line of data
        line_data = line.split()
        rating = int(line_data[0])
        qid = int(line_data[1].split(':')[1])
        features = []
        for elem in line_data[2:]:
            if '#docid' in elem:  # We reached a comment. Line done.
                break
            features.append(float(elem.split(':')[1]))
        # Creating a new data instance, inserting in the dict.
        di = DataInstance(qid, rating, features)
        if qid in data_set.keys():
            data_set[qid].append(di)
        else:
            data_set[qid] = [di]
    return data_set


class DataHolder:
    """
    A class that holds all the data in one of our sets (the training set or the test set)
    """
    def __init__(self, data_set):
        self.data_set = load_data(data_set)


def run_rank(training_set, test_set):
    # TODO: Insert the code for training and testing your ranker here.
    # Data holders for training and test set
    dh_training = DataHolder(training_set)
    dh_testing = DataHolder(test_set)

    # Creating an ANN instance - feel free to experiment with the learning rate (the third parameter).
    nn = bp.NN(46, 10, 0.001)

    # TODO: The lists below should hold training patterns in this format:
    # [(data1Features,data2Features), (data1Features,data3Features), ... , (dataNFeatures,dataMFeatures)]
    # TODO: The training set needs to have pairs ordered so the first item of the pair has a higher rating.
    training_patterns = []  # For holding all the training patterns we will feed the network
    test_patterns = []  # For holding all the test patterns we will feed the network

    for qid in dh_training.data_set.keys():
        # This iterates through every query ID in our training set
        data_instance=dh_training.data_set[qid]  # All data instances (query, features, rating) for query qid
        # TODO: Store the training instances into the training_patterns array.
        # Remember to store them as pairs, where the first item is rated higher than the second.
        # TODO: Hint: A good first step to get the pair ordering right, is to sort the instances
        # based on their rating for this query. (sort by x.rating for each x in DataInstance)

    for qid in dh_testing.data_set.keys():
        # This iterates through every query ID in our test set
        data_instance = dh_testing.data_set[qid]
        # TODO: Store the test instances into the test_patterns array, once again as pairs.
        # TODO: Hint: The testing will be easier for you if you also now order the pairs -
        # it will make it easy to see if the ANN agrees with your ordering.

    # Check ANN performance before training
    nn.count_misordered_pairs(test_patterns)
    for i in range(25):
        # Running 25 iterations, measuring testing performance after each round of training.
        # Training
        nn.train(training_patterns, iterations=1)
        # Check ANN performance after training.
        nn.count_misordered_pairs(test_patterns)

    # TODO: Store the data returned by count_misordered_pairs and plot it, showing how training and testing errors develop.

run_rank("train.txt", "test.txt")
