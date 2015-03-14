__author__ = 'eirikvageskar'
import random
import math


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

    return -(q*math.log(q, 2)+(1-q)*log((1-q), 2))


def find_true_count(examples, example_numbers):
    """
    Finds the number of true results in the subset.
    True is assumed to be the largest of the result values.

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

    return true_count



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
    pass


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
            values[v].add(e)        # Add example number e to the set of examples.
        else:
            values[v] = {e}         # If set doesn't exist, make a new one.

    return values


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
    if len(example_numbers) == 0:
        return plurality_value(examples, parent_example_numbers)

    random_index = random.sample(example_numbers, 1)[0]
    random_result = examples[random_index][-1]

    for e in example_numbers:
        if examples[e][-1] != random_result:
            break
    else:
        # If previous loop executed without breaks, all have same classification
        # Thus, return this classification
        return random_result

    if len(attribute_set) == 0:
        return plurality_value(examples, example_numbers)

    #Find argmax
    max_importance = -1
    for a in attribute_set:
        a_importance = importance(examples, example_numbers, a)
        if a_importance > max_importance:
            max_importance = a_importance
            argmax = a

    #Find values of argmax-attribute and the examples that take on those values
    vals = find_values_and_example_numbers(examples, example_numbers, argmax)

    tree = {"root_test": argmax}

    new_attribute_set = attribute_set.difference([argmax])

    for v in vals:
        tree[v] = decision_tree_learning(examples, vals[v], new_attribute_set, example_numbers, importance)

    return tree


def read_examples(filepath):
    """
    Read examples from a file and put them in a list.

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




