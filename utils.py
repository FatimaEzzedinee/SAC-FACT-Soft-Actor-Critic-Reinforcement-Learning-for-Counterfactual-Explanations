import pandas as pd
import numpy as np
from gower import *

class GenPoint(object):
    def __init__(self, name, label):
        self.features = None
        self.name = name
        self.label = label
    
    def set_position(self, values):
        self.features = values
    
    def get_position(self):
        return self.features
    
    def move(self, feature_position, perc):
      self.features[feature_position] = dec2(self.features[feature_position] + self.features[feature_position] * perc)


def minMax(data):
    return pd.Series(index=['min','max'],data=[data.min(),data.max()])


def dec2(x):
  return float("{:.8f}".format(x))


def probability(values, a, b):
  return ((a <= values) & (values <= b)).sum()/len(values)

def find_interval(np_intervals, value):
  intervals = pd.arrays.IntervalArray.from_tuples(np_intervals, closed='both')
  try:
    return np_intervals[list(intervals.contains(value)).index(True)], list(intervals.contains(value)).index(True)   #interval, index of the interval
  except:
    return (0, 0), False

def prepare_intervals(max_value, interval_width = 0.5):
  tuples = []
  for i in np.arange(0, max_value+1, interval_width):
    tuples.append((i, i+interval_width))
  return tuples

def compute_intervals_probabilities(values, np_intervals):
  probs = []
  for i in range(len(np_intervals)):
    a, b = np_intervals[i]
    probs.append(probability(values, a, b))
  return probs

def closest_interval_count(index, np_intervals, values, threshold):
  probs = compute_intervals_probabilities(values, np_intervals)
  high_probs_indexes = np.array(probs) >=threshold
  dist = np.asarray(np.where(high_probs_indexes == True)) - index
  result = np.argmin(np.absolute(dist[0]))
  return dist[0][result] + index, np.min(np.absolute(dist[0])) #index of closet interval, number of intervals in between


def gower_custom(initial_dataset, p1, p2):
   df_copy = initial_dataset.copy()
   df_copy.loc[len(df_copy)] = p1
   df_copy.loc[len(df_copy)] = p2
   return gower_matrix(df_copy)[len(df_copy)-1, len(df_copy)-2]