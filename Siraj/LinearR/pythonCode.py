from numpy import *


def run():
  points = genfromtext('data.csv', delimiter=',') #from numpy
  #hyperparameters
  #if learning rate too low, model too slow to converg. if too high, never converge
  #we don't always know, we guess and check
  learning_rate = 0.0001 
  initial_b = 0
  initial_m = 0 #will learn these values over time
  num_iterations = 1000 #small data set so we're doing low number of iterations, incorporate gpu's etc
  [b, m] = gradient_descent_runner
  print(b)
  print(m)


if__name__ = '__main__':
  run()
