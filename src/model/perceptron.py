# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from sklearn.metrics import accuracy_score
from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

        #add bias
        np.insert(self.weight, 0, np.random.rand()/10)

        np.insert(self.trainingSet.input, 0, 1, axis=1)
        np.insert(self.validationSet.input, 0, 1, axis=1)
        np.insert(self.testSet.input, 0, 1, axis=1)

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        
        # Write your code to train the perceptron here
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}").format(epoch+1, self.epochs)

            for xi, label in zip(self.trainingSet.input, self.trainingSet.label):
                error = label - self.classify(xi)
                self.updateWeights(xi, error)

            if verbose:
                accuracy = accuracy_score(self.validationSet.label, self.evaluate(self.validationSet))
                print("Accuracy on validation: {0:.2f}%").format(accuracy*100)
                print("------------------------------")

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        # Write your code to do the classification on an input image
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        self.weight += self.learningRate * input * error
         
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.dot(np.array(input), self.weight))
