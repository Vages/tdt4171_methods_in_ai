__author__ = 'eirikvageskar'
import random
import math

"""
Made for Python3
"""


def plurality_value(examples, example_numbers):
    """
    Finds the most common output value (i.e. last value) among a set of examples.

    :param examples: The complete set of examples.
    :param example_numbers: The examples to be examined.
    :return: The most common result.
    """

    counts = {}

    for i in example_numbers:
        result = examples[i][-1]
        if result in counts:
            counts[result] += 1
        else:
            counts[result] = 1

    return max(counts, key=lambda x: counts[x])


def boolean_entropy(q):
    """
    Boolean entropy for a variable with probability q of being true

    :param q: The probability of the variable being true
    :return: Boolean entropy
    """

    try:
        return -(q*math.log2(q)+(1-q)*math.log2(1-q))
    except ValueError:
        return 0


def find_true_count(examples, example_numbers):
    """
    Finds the number of true results in the subset.
    True is assumed to be the largest of the result values.

    :param examples: The entire set of examples.
    :param example_numbers: The subset in question.
    :return: The proportion of true variables.
    """

    true_value = max(examples, key=lambda x: x[-1])[-1]

    true_count = 0
    for e in example_numbers:
        if examples[e][-1] == true_value:
            true_count += 1

    return true_count


def remainder(examples, example_numbers, a):
    """
    Finds expected entropy after testing attribute a.

    :param examples: The entire set of examples.
    :param example_numbers: The subset in question.
    :param a: The attribute in question.
    :return: Expected remaining entropy.
    """

    values = find_values_and_example_numbers(examples, example_numbers, a)

    remainder_sum = 0
    p_plus_n = len(example_numbers)
    for k in values:
        pk_plus_nk = len(values[k])
        pk = find_true_count(examples, values[k])
        boolean_entropy_of_k_set = boolean_entropy(pk/pk_plus_nk)
        remainder_sum += (pk_plus_nk/p_plus_n)*boolean_entropy_of_k_set

    return remainder_sum


def random_importance(examples, example_numbers, a):
    """
    A sheep function in wolf's clothing: Returns a random number.

    :param examples:
    :param example_numbers:
    :param a:
    :return:
    """
    return random.random()


def information_gain(examples, example_numbers, a):
    """
    Finds the information gain of splitting the subset using attribute a.

    :param examples: The entire set of examples.
    :param example_numbers: The subset in question.
    :param a: The attribute in question.
    :return: Information gain.
    """

    probability_of_true = find_true_count(examples, example_numbers)/len(example_numbers)
    b = boolean_entropy(probability_of_true)

    remainder_of_a = remainder(examples, example_numbers, a)

    return b - remainder_of_a


def find_values_and_example_numbers(examples, example_numbers, attribute):
    """
    Finds the possible values of attribute a in subset of examples given by example numbers

    :param examples: The examples
    :param example_numbers: Example indices
    :param attribute: The attribute to be examined
    :return: A dictionary containing attributes and sets of examples
    """
    values = {}
    
    for e in example_numbers:       # For every example number in questien
        v = examples[e][attribute]
        if v in values:
            values[v].add(e)        # Add example number e to the set of training_set.
        else:
            values[v] = {e}         # If set doesn't exist, make a new one.

    return values


def have_same_classification(examples, example_numbers):
    """
    Helper method that checks if all examples in the subset have the same classification

    :param examples:
    :param example_numbers:
    :return:
    """
    random_index = random.sample(example_numbers, 1)[0]
    random_result = examples[random_index][-1]

    for e in example_numbers:  # Check if all examples have same classification
        if examples[e][-1] != random_result:
            return False

    return random_result


def decision_tree_learning(examples, example_numbers, attribute_set, parent_example_numbers, importance):
    """
    Returns a decision tree.
    Builds on fig. 18.5 from Artificial Intelligence: A modern approach.

    :param examples: The complete set of examples to work on.
    :param example_numbers: The indices of the examples to be examined.
    :param attribute_set: The set of attributes still to be decided on.
    :param parent_example_numbers: The example_numbers of this branch's parent.
    :param importance: Function used to judge importance of an attribute.
    :return:
    """

    if len(example_numbers) == 0:  # Examples is empty
        return plurality_value(examples, parent_example_numbers)

    same_classification_check = have_same_classification(examples, example_numbers)
    if same_classification_check:
        return same_classification_check

    if len(attribute_set) == 0:  # Attribute set is empty
        return plurality_value(examples, example_numbers)

    # Find argmax
    max_importance = -1

    for a in attribute_set:
        a_importance = importance(examples, example_numbers, a)
        if a_importance > max_importance:
            max_importance = a_importance
            argmax = a

    # Construct a new tree to be returned
    tree = {"root_test": argmax}

    # Find values of argmax-attribute and the training_set that take on those values
    values = find_values_and_example_numbers(examples, example_numbers, argmax)

    new_attribute_set = attribute_set.difference([argmax])

    for v in values:
        tree[v] = decision_tree_learning(examples, values[v], new_attribute_set, example_numbers, importance)

    return tree


def read_examples(file_path):
    """
    Read examples from a file and put them in a list.

    :param file_path: Filepath or filename.
    :return: Example list
    """
    f = open(file_path)
    examples = []
    for line in f:
        examples.append(line.split())

    return examples


def classify(decision_tree, example):
    """
    Classifies the given specimen according to the decision tree.

    :param decision_tree: A decision tree (or leaf node)
    :param example: The specimen
    :return: Classification
    """

    if type(decision_tree) is not dict:  # We have reached a leaf node
        return decision_tree

    test_attribute = decision_tree["root_test"]
    return classify(decision_tree[example[test_attribute]], example)

if __name__ == "__main__":
    training_set = read_examples("data/training.txt")
    training_numbers_set = set([i for i in range(len(training_set))])

    number_of_attributes = len(training_set[0])-1
    attribute_set = set([i for i in range(number_of_attributes)])

    good_decision_tree = decision_tree_learning(training_set, training_numbers_set, attribute_set, None, information_gain)
    bad_decision_tree = decision_tree_learning(training_set, training_numbers_set, attribute_set, None, random_importance)
    test_examples = read_examples("data/test.txt")
    # test_numbers_set = set([i for i in range(len(test_examples))])

    classifications = []

    for example in test_examples:
        temp = [example[-1]]
        temp.append(classify(bad_decision_tree, example))
        temp.append(classify(bad_decision_tree, example))

    for i in range(len(classifications)):
        print(str(i) + ". Real: " + classifications[i][0] + ", Good: " + classifications[i][1] + ", Bad: " + classifications[i][2])