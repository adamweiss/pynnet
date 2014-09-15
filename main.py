import pynnet

""" Main
Loop through training 10 times and print the results
"""
for i in range(10):
  print "******************** Run: ", i, "**************************"
  n = pynnet.NN((9,9,1), -0.4, .2, sample_size = 0.66)
  n.load_training_data("breast_cancer_data.csv")
  n.ffbp_train(error_delta = 0.00000000000001, max_epochs = 500)

