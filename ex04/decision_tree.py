__author__ = 'eirikvageskar'
import random
import math
import pydot

"""
This file solves Exercise 4 given in Artificial Intelligence Methods (TDT4171) at NTNU in march 2015.
The task is implementing a Decision Tree Learning algorithm given in Artificial Intelligence, a Modern Approach.

This is realized with the decision_tree_learning algorithm at its core, as well as a few helper methods.

The file is made for Python3.
"""


def plurality_value(examples, example_numbers):
    """
    Finds the most common classification (i.e. value at last index) among a set of examples.

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
    except ValueError:  # Happens if q is 1 or 0, which causes a problem with the logarithm function.
        return 0


def find_true_count(examples, example_numbers):
    """
    Finds the number of specimen in the subset whose classification is True.
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

    :param examples: Argument added to make it replaceable with information_gain.
    :param example_numbers: Argument added to make it replaceable with information_gain.
    :param a: Argument added to make it replaceable with information_gain.
    :return: A random number between 0 and 1.
    """
    return random.random()


def information_gain(examples, example_numbers, a):
    """
    Finds the information gain from splitting the subset using attribute a.

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
    Finds the possible values of attribute a in subset of examples given by example numbers.

    :param examples: The entire set of examples.
    :param example_numbers: The subset in question.
    :param attribute: The attribute to be examined.
    :return: A dictionary containing attributes and sets of examples.
    """
    values = {}

    for ex in examples:
        val = ex[attribute]
        if val not in values:
            values[val] = set()
    
    for e in example_numbers:  # For every example number in question
        v = examples[e][attribute]
        values[v].add(e)  # Add example number e to the set of examples

    return values


def have_same_classification(examples, example_numbers):
    """
    Helper method that checks if all examples in the subset have the same classification.

    :param examples: The entire set of examples.
    :param example_numbers: The subset in question.
    :return:
    """
    random_index = random.sample(example_numbers, 1)[0]
    random_result = examples[random_index][-1]

    for e in example_numbers:  # Check if all examples have same classification
        if examples[e][-1] != random_result:
            return False

    return random_result  # No longer as random; we now know that all examples in subset share the value


def decision_tree_learning(examples, example_numbers, attribute_set, parent_example_numbers, importance):
    """
    Returns a decision tree.
    Builds on fig. 18.5 from Artificial Intelligence: A modern approach.

    :param examples: The complete set of examples to work on.
    :param example_numbers: The indices of the examples to be examined.
    :param attribute_set: The set of attributes still to be decided on.
    :param parent_example_numbers: The example_numbers of this branch's parent.
    :param importance: Function used to judge importance of an attribute.
    :return: The decision tree.
    """

    if len(example_numbers) == 0:  # Example subset is empty
        return plurality_value(examples, parent_example_numbers)

    same_classification_check = have_same_classification(examples, example_numbers)
    if same_classification_check:  # All examples have the same classification
        return same_classification_check

    if len(attribute_set) == 0:  # Attribute set is empty
        return plurality_value(examples, example_numbers)

    max_importance = -1  # Dummy importance value

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

    for v in values:  # Construct subtrees for every possible value argmax can take on
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


def test_for_accuracy(decision_tree, test_set):
    """Tests the accuracy for the given decision tree on the given test set.

    :param decision_tree:
    :param test_set:
    :return:
    """

    results = []
    for item in test_set:
        results.append((item[-1], classify(decision_tree, item)))

    erroneous_indices = []
    for i in range(len(results)):
        x, y = results[i][0], results[i][1]
        if x != y:
            erroneous_indices.append(i)

    return len(erroneous_indices)/len(test_set), erroneous_indices


def build_graph(graph, node, address=""):
    """Builds a graph of the decision tree in the dot language recursively.
    This depends on the pydot library (pydot3k), which again depends on GraphViz.

    :param graph: The graph we are working with. Must be declared from the outside.
    :param node: The node being examined (a dictionary).
    :param address: The address of this node (sequence of attribute values thus far; a unique address)
    :return: The pydot Node being that has been constructed for this dictionary node. Necessary for recursion.
    """

    this_node = pydot.Node(address, label=node["root_test"])  # Construct this node
    graph.add_node(this_node)  # Add it to the graph
    for key in node:  # For all keys
        if key == 'root_test':  # ... except the one telling us which attribute is being tested
            continue
        value = node[key]
        child_address = address+"."+str(key)  # Make a unique address for the child node
        if type(value) is dict:
            # Build the graph for the child and then connect the parent node to it
            graph.add_edge(pydot.Edge(this_node, build_graph(graph, value, child_address), label=key))
        else:
            # The child is a classification leaf node. It needs no subgraph building
            classification_node = pydot.Node(child_address, label=value, shape="rectangle")
            graph.add_node(classification_node)
            graph.add_edge(pydot.Edge(this_node, classification_node, label=key))

    return this_node


if __name__ == "__main__":
    training_set = read_examples("data/training.txt")
    training_numbers_set = set([i for i in range(len(training_set))])

    number_of_attributes = len(training_set[0])-1
    attribute_set = set([i for i in range(number_of_attributes)])

    test_examples = read_examples("data/test.txt")

    number_of_runs = 10

    good_accuracies = []
    bad_accuracies = []

    for i in range(number_of_runs):
        good_decision_tree = decision_tree_learning(training_set, training_numbers_set, attribute_set, None, information_gain)
        bad_decision_tree = decision_tree_learning(training_set, training_numbers_set, attribute_set, None, random_importance)
        good_accuracies.append(test_for_accuracy(good_decision_tree, test_examples))
        bad_accuracies.append(test_for_accuracy(bad_decision_tree, test_examples))

    print("Information gain accuracies and erroneous indices")
    for item in good_accuracies:
        print(item)

    print("\nRandom importance accuracies and erroneous indices")
    for item in bad_accuracies:
        print(item)

    good_graph = pydot.Dot(graph_type='digraph')
    build_graph(good_graph, good_decision_tree)
    good_graph.write('good_decision_tree.eps', format="eps")

    bad_graph = pydot.Dot(graph_type='digraph')
    build_graph(bad_graph, bad_decision_tree)
    bad_graph.write('bad_decision_tree.eps', format='eps')