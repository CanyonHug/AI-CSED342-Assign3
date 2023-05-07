#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################


def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
        # step size = 1, wieght setting is 0 vector
        # after learning "preety good" : w = [1, 1, 0, 0, 0, 0]
        # after learning "bad plot" : w = [1, 1, -1, -1, 0, 0]
        # after learning "not good" : w = [1, 0, -1, -1, -1, 0]
        # after learning "preety scenery" : w = [1, 0, -1, -1, -1, 0]
    return {'pretty' : 1, 'good' : 0, 'bad' : -1, 'plot' : -1, 'not' : -1, 'scenery' : 0}
    # END_YOUR_ANSWER


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    weight = collections.defaultdict(int)
    for word_count in x.split():
        weight[word_count] += 1
    return weight
    # END_YOUR_ANSWER


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    # gradient of loss is -p'/p
    # p' = sig*(1-sig)*y   -> -p'/p = -y*(1-p)
    def predict(x):
        if dotProduct(weights, featureExtractor(x)) > 0:
            return 1
        else:
            return -1

    for i in range(numIters):
        for data in trainExamples:
            x,y = data
            featured=featureExtractor(x)
            weighted=dotProduct(weights,featured) # w * featured
            if y == 1:
                p = sigmoid(weighted)
            else:
                p = 1 - sigmoid(weighted)
            increment(weights, eta*y*(1-p), featured)
        print("Iteration:%s, Training error:%s, Test error:%s"%(i,evaluatePredictor(trainExamples,predict),evaluatePredictor(testExamples,predict)))
  
    # END_YOUR_ANSWER
    return weights


############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    phi = {}
    word_count = x.split()
    for i in range(len(word_count)):
        if i != len(word_count)-1:
            phi[(word_count[i], word_count[i+1])] = phi.get((word_count[i], word_count[i+1]), 0) + 1
        phi[word_count[i]] = phi.get(word_count[i], 0) + 1
    phi[(word_count[-1], '</s>')] = 1
    phi[('<s>', word_count[0])] = 1
    # END_YOUR_ANSWER
    return phi
