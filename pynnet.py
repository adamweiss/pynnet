##################################################
# Author: Adam Weiss
##################################################

from numpy import *
import csv
import time
import copy

@vectorize
def sigmoid(x):
  if(x > 50):
    return 0
  elif (x < -50):
    return 1
  else:
    return (1/(1+math.exp(-1*(x))))

"""
calculate mean squared error
"""
def mse(d, x):
  return ((d-x)**2)/2

def zeroValues(perceptronList):
  activity = []
  activation = []
  for i in range(1, len(perceptronList)):
    activity.append(zeros(perceptronList[i]))
    activation.append(zeros(perceptronList[i]))
  return activity, activation

def calculateActivation(data, weights, perceptronList):
  activity, activation = zeroValues(perceptronList)

  # calculate hidden layer first
  activity[0] = dot(data[:-1],weights[0])

  # set bias
  activity[0][-1] = 1

  activation[0] = sigmoid(activity[0])
  for i in range(1,len(activation)):
    activity[i] = dot(activation[i-1], weights[i])
    activation[i] = sigmoid(activity[i])

  return activation[-1][-1]

"""
Neural Network class
"""
class NN(object):

  """
  perceptronList is a tuple of dimensions (inputs, hidden layer, output layer)
  """
  def __init__(self, perceptronList, bias = -0.6, rate = 0.1, sample_size = 0.5):
    self.perceptronList = perceptronList
    self.rate = rate
    self.bias = bias
    self.randomizeWeights()
    self.error = 999999999999999999999999
    self.iterations = 0
    self.sample_size = sample_size
    self.resetValues()

  """
  will reset activity and activation values
  """
  def resetValues(self):
    self.activity = []
    self.activation = []
    self.delta = []
    self.error = 9999999999999999999999999
    for i in range(1,len(self.perceptronList)):
      self.activity.append(zeros(self.perceptronList[i]))
      self.activation.append(zeros(self.perceptronList[i]))
      self.delta.append(zeros(self.perceptronList[i]))


  """
  initialize a list of numpy arrays with random weights given dimensions perceptronList
  """
  def randomizeWeights(self):
    self.weights = []
    for i in range(len(self.perceptronList)-1):
      if(self.perceptronList[i] != 0 and self.perceptronList[i+1] != 0):
        # use bipolar values instead of values greater than zero
        self.weights.append(random.randn(self.perceptronList[i], self.perceptronList[i+1]))

    self.weights[-1][-1] = self.bias

  """
  initialize a list of numpy arrays with random bipolar normally distributed weights given dimensions perceptronList
  """
  def randomizeBipolarWeights(self):
    self.weights = []
    for i in range(len(self.perceptronList)-1):
      if(self.perceptronList[i] != 0 and self.perceptronList[i+1] != 0):
        self.weights.append(random.randn(self.perceptronList[i], self.perceptronList[i+1]))


  """
  will generate weights of 1 for standardized debugging.  not typically used
  """
  def defaultWeights(self):
    self.weights = []
    for i in range(len(self.perceptronList)-1):
      if(self.perceptronList[i] != 0 and self.perceptronList[i+1] != 0):
        self.weights.append(ones((self.perceptronList[i], self. perceptronList[i+1])))
 
  """
  load training data.  if scale_data is set to 1, scale the training data
  to between 0 and 1
  """
  def load_training_data(self, filename, scale_data = 1):
    self.data = genfromtxt(filename, delimiter=",")
    print "Loading file: ", filename

    if(scale_data == 1):
      print "Scaling loaded data from 0 to 1\n"
      for i in range(self.data.shape[1]-1):
        self.data[:,i] = self.data[:,i] / max(self.data[:,i])

  """
  Calculates the activation function given data
  """
  def calculateActivation(self, data):
    self.activity[0] = dot(data[:-1],self.weights[0])

    self.activation[0] = sigmoid(self.activity[0])
    for i in range(1,len(self.activation)):
      self.activity[i] = dot(self.activation[i-1],self.weights[i])
      self.activation[i] = sigmoid(self.activity[i])

    return self.activation[-1][-1]

  """
  Feed forward back propogation train
  """
  def ffbp(self, data = None):
    if(data is None):
      data = ones(self.perceptronList[0])

    # calculate hidden layer first
    self.activity[0] = dot(data[:-1],self.weights[0])

    # set bias
    self.activity[0][-1] = 1

    self.activation[0] = sigmoid(self.activity[0])

    #calculate remaining layers
    for i in range(1,len(self.activation)):
      self.activity[i] = dot(self.activation[i-1],self.weights[i])
      self.activation[i] = sigmoid(self.activity[i])

    #calculate delta for output layer
    self.delta[-1] = self.activation[-1] * (1 - self.activation[-1]) * (data[-1] - self.activation[-1])
    # calculate delta for hidden layer(s)

    for i in range(len(self.activation)-2, -1, -1):
      self.delta[i] = sum((self.activation[i] * (1-self.activation[i])) * self.delta[i+1] * self.weights[i], 0)

    self.error = (data[-1] - self.activation[-1])**2/2

    #update output layer weights
    for i in range(1, len(self.weights)):
      #import pdb; pdb.set_trace()
      self.weights[i] = self.weights[i] + (self.activation[i-1].reshape(self.activation[i-1].size, 1) * self.delta[i] * self.rate)

    #update weights for input/hidden layer
    self.weights[0] = self.weights[0] + (data[:-1].reshape(self.perceptronList[0],1) * self.delta[0]) * self.rate

    # reset bias weight
    self.weights[-1][-1] = self.bias

  """
  this will implement a simple random training algorithm.  it will use
  normally distributed random numbers (mean = 0) to modify all weights.
  Training data is run through the old and new weights.
  If a modification lowers the MSE, keep the new weights
  The randomize flag will reset weights to random
  """
  def random_search_train(self, randomize = 1, sampling = None, max_epochs = 3000, error_delta=0.0000000000000000000000001):
    print "Performing random search training algorithm with the following parameters"
    print "{0:<20} {1:>5}".format("Max epochs:", max_epochs)
    print "{0:<20} {1:>5}\n\n".format("Random weights:", randomize)

    if(randomize == 1):
      self.randomizeWeights()

    # generate new samples if needed
    if(sampling == None):
      sampling = random.random_integers(0, len(self.data)-1, len(self.data)*self.sample_size)
      self.sampling = unique(sampling)

    # initialize values
    epoch = 0
    SamplingError = 999999999999
    newSamplingError = 0
    newWeights = copy.deepcopy(self.weights)
    samplingErrors = 0

    for i in self.sampling:
      activation = self.calculateActivation(self.data[i])
      error = mse(self.data[i][-1], activation)
      samplingErrors += error

    SamplingError = samplingErrors/len(self.sampling)

    # perform training until we hit max_epochs
    while(epoch < max_epochs):
      newSamplingErrors = 0

      # Will compare all array entries, return true if they're the same
      arrays_equal = False
      for i in range(len(self.weights)):
        if(array_equal(newWeights[i], self.weights[i])):
          arrays_equal = True
        else:
          arrays_equal = False
          break

      # randomly update weights according to normal distribution with mean of
      # 0 and std of 0.2
      for i in range(len(self.perceptronList)-1):
        if(self.perceptronList[i] != 0 and self.perceptronList[i+1] != 0):
          newWeights[i] = newWeights[i] + (random.normal(0, 0.2, (self.perceptronList[i], self.perceptronList[i+1])))

      #set bias
      newWeights[-1][-1][-1] = self.bias

      # Resample with new weights
      for i in self.sampling:
        newActivation = calculateActivation(self.data[i], newWeights, self.perceptronList)
        newError = mse(self.data[i][-1], newActivation)
        newSamplingErrors += newError

      newSamplingError = newSamplingErrors / len(self.sampling)

      # if the average error with the new weights is reduced, update the weights
      if(newSamplingError < SamplingError):
        self.weights = copy.deepcopy(newWeights)
        SamplingError = newSamplingError
        print newSamplingError
      else:
        newWeights = copy.deepcopy(self.weights)
      epoch += 1

    self.determine_accuracy(self.data, self.sampling)

  """
  scale_data == 1 will auto scale training data.  0 will leave as is
  """
  def ffbp_train(self, sampling = None, randomize = 1, error_delta=0.00000000000001, max_epochs = 2000):
    #data = csvToIntList(filename)
    print "Performing FFBP training algorithm with the following parameters"
    print "{0:<20} {1:>5}".format("Max epochs:", max_epochs)
    print "{0:<20} {1:>5}".format("Min error delta:", error_delta)
    print "{0:<20} {1:>5}\n\n".format("Random weights:", randomize)

    if(randomize == 1):
      self.randomizeWeights()

    # generate new sampling data if sampling == None
    if(sampling == None):
      # sample half of the data
      sampling = random.random_integers(0, len(self.data)-1, len(self.data)*self.sample_size)
      self.sampling = unique(sampling)

    # calculate average error of 1 epoch
    error = 0
    newerror = 100000
    epoch = 0
    while(((newerror-error)**2/2 > error_delta) and epoch < max_epochs):
      error = newerror
      errors = []
      for i in self.sampling:
        self.ffbp(self.data[i])
        errors.append(self.error)
      newerror = average(errors)
      print newerror
      epoch += 1

    self.determine_accuracy(self.data, self.sampling)

  """
  Determine specificity and sensitivity of training and untrained data with
  current network
  """
  def determine_accuracy(self, data, sampling):
    sampling.sort()
    samples = sampling.tolist()

    positive = []
    negative = []

    # trained data accuracy:
    for i in samples:
      # if this is a sampling data point, skip it and get rid of sampling data point
      activation = self.calculateActivation(data[i])
      if(data[i][-1] == 0):
        negative += [activation]
      else:
        positive += [activation]

    print "\nTrained samples: ", len(sampling)
    print "Untrained samples: ", len(data) - len(sampling), " \n"

    print "\n\n", "Trained data results".center(54), "\n"
    if(len(negative) > 0):
      print "{0:>25}    {1:<10}".format("Class 0 average:", mean(negative))
      print "{0:>25}    {1:<10}".format("Class 0 min:" , min(negative))
      print "{0:>25}    {1:<10}".format("Class 0 max:" , max(negative))
    else:
      print "No Class 0 samples"

    if(len(positive) > 0):
      print "{0:>25}    {1:<10}".format("Class 1 average:" , mean(positive))
      print "{0:>25}    {1:<10}".format("Class 1 min:" , min(positive))
      print "{0:>25}    {1:<10}".format("Class 1 max:" , max(positive))
    else:
      print "No Class 1 samples"

    threshold = -1*self.bias
    if(len(negative) > 0):
      tn = len([i for i in negative if i < threshold])
      fp = len([i for i in negative if i >= threshold])
      trainedSpecificity = tn / float(tn + fp)
    else:
      tn,fp,trainedSpecificity = 0,0,0

    if(len(positive) > 0):
      tp = len([i for i in positive if i >= threshold])
      fn = len([i for i in positive if i < threshold])
      trainedSensitivity = tp / float(tp + fn)
    else:
      tp,fn,trainedSensitivity = 0,0,0

    trainedAccuracy = (tp + tn)/ float(len(sampling))
    print "{0:>25}    {1:<10}".format("Threshold:" , threshold)
    print "{0:>25}    {1:<10}".format("True positive:" , tp)
    print "{0:>25}    {1:<10}".format("False positive:", fp)
    print "{0:>25}    {1:<10}".format("True negative:", tn)
    print "{0:>25}    {1:<10}".format("False negative:", fn)
    print "{0:>25}    {1:<10}".format("Trained sensitivity:" , trainedSensitivity)
    print "{0:>25}    {1:<10}".format("Trained specificity:" , trainedSpecificity)
    print "{0:>25}    {1:<10}".format("Trained accuracy:" , trainedAccuracy)

    positive = []
    negative = []
    for idx, value in enumerate(data):
      # if this is a sampling data point, skip it and get rid of sampling data point
      if(len(samples) > 0 and idx == samples[0]):
        samples.pop(0)
        continue
      activation = self.calculateActivation(value)
      if(value[-1] == 0):
        negative += [activation]
      else:
        positive += [activation]

    print "\n\n", "Non-trained data accuracy".center(54), "\n"

    if(len(negative) > 0):
      print "{0:>25}    {1:<10}".format("Class 0 average:", mean(negative))
      print "{0:>25}    {1:<10}".format("Class 0 min:" , min(negative))
      print "{0:>25}    {1:<10}".format("Class 0 max:" , max(negative))
    else:
      print "No Class 0 samples"

    if(len(positive) > 0):
      print "{0:>25}    {1:<10}".format("Class 1 average:" , mean(positive))
      print "{0:>25}    {1:<10}".format("Class 1 min:" , min(positive))
      print "{0:>25}    {1:<10}".format("Class 1 max:" , max(positive))
    else:
      print "No Class 1 samples"

    threshold = -1*self.bias
    if(len(negative) > 0):
      tn = len([i for i in negative if i < threshold])
      fp = len([i for i in negative if i >= threshold])
      untrainedSpecificity = tn / float(tn + fp)

    if(len(positive) > 0):
      tp = len([i for i in positive if i >= threshold])
      fn = len([i for i in positive if i < threshold])
      untrainedSensitivity = tp / float(tp + fn)

    untrainedAccuracy = (tp + tn)/ float(len(data) - len(sampling))
    print "{0:>25}    {1:<10}".format("Threshold:" , threshold)
    print "{0:>25}    {1:<10}".format("True positive:" , tp)
    print "{0:>25}    {1:<10}".format("False positive:", fp)
    print "{0:>25}    {1:<10}".format("True negative:", tn)
    print "{0:>25}    {1:<10}".format("False negative:", fn)
    print "{0:>25}    {1:<10}".format("Untrained sensitivity:" , untrainedSensitivity)
    print "{0:>25}    {1:<10}".format("Untrained specificity:" , untrainedSpecificity)
    print "{0:>25}    {1:<10}".format("Untrained accuracy:" , untrainedAccuracy)

    print "\n"

