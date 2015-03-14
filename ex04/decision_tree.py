__author__ = 'eirikvageskar'
import random
import math


def plurality_value(examples, example_numbers):
    """Finds the most common output value (i.e. last value) among a set of examples.

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


def decision_tree_learning(examples, example_numbers, attribute_set, parent_example_numbers):
def find_true_proportion(examples, example_numbers):
    """Finds the proportion of true results in the subset.

    :param examples: The entire set of examples.
    :param example_numbers: The subset in question.
    :return: The proportion of true variables.
    """

    # The value for "True" is expected to be the highest of the result values.
    # This function should never be run if there is only one kind of value in the results.

    true_value = max(examples, key=lambda x: x[-1])[-1]

    true_count = 0
    for e in example_numbers:
        if examples[e][-1] == true_value:
            true_count += 1

    return true_count/len(example_numbers)
    """Returns a decision tree.

    Builds on fig. 18.5 from Artificial Intelligence: A modern approach.

    :param examples: The complete set of examples to work on.
    :param example_numbers: The indices of the examples to be examined.
    :param attribute_set: The set of attributes still to be decided on.
    :param parent_example_numbers: The example_numbers
    :return:
    """
    if len(example_numbers) == 0:
        return plurality_value(examples, parent_example_numbers)
    if len(attribute_set) == 0:
        return plurality_value(examples, example_numbers)

    max_importance = -1
    for a in attribute_set:
        a_importance = importance(a, example_numbers)
        if a_importance > max_importance:
            max_importance = a_importance
            argmax = a

    tree = {"root_test": argmax}

    #We need some recursive loop here.

    return tree


def read_examples(filepath):
    """Read examples from a file and put them in a list.

    :param filepath: Filepath or filename.
    :return: Example list
    """
    f = open(filepath)
    examples = []
    for line in f:
        examples.append(line.split())

    return examples

if __name__ == "__main__":
    exes = read_examples("data/test.txt")
    ex_nos = [i for i in range(len(exes))]

    ex_nos = set(ex_nos)

    print plurality_value(exes, ex_nos)




