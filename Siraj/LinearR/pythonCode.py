#yt link (https://www.youtube.com/watch?v=XdM6ER7zTLk&list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3&index=2)
from numpy import *

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
  b = starting_b
  m = starting_m

  for i in range(num_iterations):
    b, m = step_gradient(b, m, array(points), learning_rate) #take x, y value and feed it into array


def run():
  points = genfromtext('data.csv', delimiter=',') #from numpy
  #hyperparameters
  #if learning rate too low, model too slow to converg. if too high, never converge
  #we don't always know, we guess and check
  learning_rate = 0.0001 
  initial_b = 0
  initial_m = 0 #will learn these values over time
  num_iterations = 1000 #small data set so we're doing low number of iterations, incorporate gpu's etc
  [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
  print(b)
  print(m)


if__name__ = '__main__':
  run()
