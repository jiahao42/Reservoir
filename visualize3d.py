#! /usr/bin/env python3
# -*- coding: utf-8

import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import json

if __name__ == '__main__':
  if (len(sys.argv) == 1):
    raise RuntimeError('No filename provided!')
  filename = sys.argv[1]
  with open(filename) as input_file:
    data = input_file.readlines()
  print (''.join(str(e) for e in data[1:30]))
  # Read JSON config
  # config = json.loads(''.join(str(e) for e in data[1:30]), object_pairs_hook = OrderedDict)
  # Filt all the actual data
  data = list(filter(lambda x: x[0] == 'N', data))
  data = list(filter(lambda x: x[4:6] != '13', data)) # temperory hack
  data = list(map(lambda x: x.strip().split(','), data))
  nodes = list(map(lambda x: int(x[0][4:]), data))
  degrees = list(map(lambda x: float(x[1][5:]), data))
  RMS = list(map(lambda x: float(x[2][7:]), data))
  data = list(zip(nodes, degrees, RMS))
  X = np.zeros((2, len(nodes)))
  X[0][:] = nodes
  X[1][:] = RMS
  print (str(len(X[0])) + ' ' + str(len(X[1])))
  print (X[0][0:100])
  print (X[1][0:100])
  Y = np.zeros((2, len(nodes)))
  Y[0][:] = nodes
  Y[1][:] = RMS
  print (str(len(Y[0])) + ' ' + str(len(Y[1])))
  print (Y[0][0:100])
  print (Y[1][0:100])
  Z = np.zeros((2, len(nodes)))
  Z[0][:] = degrees
  Z[1][:] = nodes
  print (str(len(Z[0])) + ' ' + str(len(Z[1])))
  print (Z[0][0:100])
  print (Z[1][0:100])
  data = sorted(data, key = lambda x: (x[0]))
  # list(map(print, data))
  # degree_function = config["reservoir"]["degree_function"].split(':')[1]
  end = len(nodes)
  # plt.text(2000, 1.5, degree_function, size=10, ha="center")
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  # plt.title('D = ' + degree_function)
  ax.scatter(nodes, degrees, RMS, c = 'b')
  # plt.plot(nodes[0:end], RMS[0:end])
  plt.xlabel('Nodes')
  plt.ylabel('Degree')
  plt.tight_layout()
  plt.show()