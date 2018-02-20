#! /usr/bin/env python3
# -*- coding: utf-8

import numpy as np
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
  print (''.join(str(e) for e in data[1:27]))
  # Read JSON config
  config = json.loads(''.join(str(e) for e in data[1:27]), object_pairs_hook = OrderedDict)
  # Filt all the actual data
  data = list(filter(lambda x: x[0] == 'N', data))
  data = list(map(lambda x: x.strip().split(','), data))
  nodes = list(map(lambda x: int(x[0][4:]), data))
  degrees = list(map(lambda x: float(x[1][5:]), data))
  RMS = list(map(lambda x: float(x[2][7:]), data))
  data = list(zip(nodes, degrees, RMS))
  list(map(print, data))
  degree_function = config["reservoir"]["degree_function"].split(':')[1]
  # plt.text(2000, 1.5, degree_function, size=10, ha="center")
  plt.title('D = ' + degree_function)
  plt.plot(nodes, RMS)
  plt.xlabel('Nodes')
  plt.ylabel('RMS error')
  plt.tight_layout()
  plt.show()