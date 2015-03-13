__author__ = 'eirikvageskar'


def decision_tree_learning(examples, attribute_set, parent_examples):
    """Returns a decision tree.

    Builds on fig. 18.5 from Artificial Intelligence: A modern approach.

    :param examples: The set of examples to be examined.
    :param attribute_set: The set of attributes still to be decided on.
    :param parent_examples: The examples
    :return:
    """
    if len(examples) == 0:
        return plurality_value(parent_examples)
    if len(attribute_set) == 0:
        return plurality_value(examples)

    max_importance = -1
    for a in attribute_set:
        a_importance = importance(a, examples)
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




