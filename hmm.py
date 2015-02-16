__author__ = 'eirikvageskar'

import numpy as np

def normalize(a):
    """Normalizes a numpy array so that the sum of the elements is 1.

    :param a: A numpy array
    :return: The normalized array
    """
    return a*1/a.sum()

def forward(t, o, f):
    """Computes one HMM forward step and normalizes the vector.

    :param t: Transition model
    :param o: An observation matrix for the transition model
    :param f: The last forward value
    :return: The forward value
    """

    temp = o*t.transpose()*f
    return normalize(temp)

def backward(t, o, b):
    """Computes one HMM backward step

    :param t: Transition model
    :param o: An observation matrix for the transition model
    :param b: The last backward value
    :return: The backward value
    """

    return t*o*b

def forward_with_observations(t, obs_dict, f, observations):
    """

    :param t: Transition model
    :param obs_dict: A dictionary of observation matrices for different observation values
    :param f: First f-value
    :param observations: The observations
    :return: The last f-value
    """

    for i in range(len(observations)):
        f = forward(t, obs_dict[observations[i]], f)
        if show_normalized:
            print "\nf1:" + str(i+1) + "\n", f

    return f

def forward_backward(t, obs_dict, f, observations):
    """An implementation of the forward-backward algorithm on page, fig 14.4 on p. 576 of Artificial Intelligence:
    A Modern Approach (3rd edition).

    :param t: Transition model
    :param obs_dict: A dictionary of observation matrices for different observation values
    :param f: First f-value
    :param observations: The observations
    :return: A list of smoothed values for
    """

    fv = [f]

    for i in range(len(observations)):
        fv.append(forward(t, obs_dict[observations[i]], fv[i]))

    sv = [np.matrix("1; 1")]*(len(fv)-1)

    b = np.matrix("1; 1")

    for i in range(len(sv)):
        j = -(i+1)
        sv[j] = normalize(np.multiply(fv[j], b))

        b = backward(t, obs_dict[observations[j]], b)
        if show_backward:
            print "\nb" + str(len(sv)-i) + ":" + str(len(sv)) + "\n", b

    return sv



if __name__ == "__main__":
    print "Part B\nTask 1"

    t = np.matrix('0.7 0.3; 0.3 0.7')
    f = np.matrix('0.5; 0.5')
    ot = np.matrix('0.9 0; 0 0.2')
    of = np.matrix('0.1 0; 0 0.8')
    obs_dict = {'t':ot, 'f':of}

    show_normalized = False

    fwo1 = forward_with_observations(t, obs_dict, f, ['t', 't'])
    print "\nProb. vector on day 2, given (t, t):\n", fwo1

    print "\nTask2"

    show_normalized = True

    fwo2 = forward_with_observations(t, obs_dict, f, ['t', 't', 'f', 't', 't'])
    print "\nProb. of rain on day five given (t, t, f, t, t):\n", fwo2[0, 0]

    print "\nPart C\nTask 1"

    show_backward = False

    fb1 = forward_backward(t, obs_dict, f, ['t', 't'])
    print "\nValue of X1 after smoothing:\n", fb1[0]

    print "\nTask 2"

    show_backward = True

    fb2 = forward_backward(t, obs_dict, f, ['t', 't', 'f', 't', 't'])
    print "\nProb of rain in X1 after long smoothing:\n", fb2[0]