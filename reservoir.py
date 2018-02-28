#! /usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import networkx as nx
import json
import atexit
import os.path
from decimal import Decimal
from collections import OrderedDict
import datetime
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt

# if config file not exists, use this default config
default_config = """{
  "input": {
    "nodes": 2,
    "functions": 
      [
        "lambda x: np.sin(128 * np.pi * x)",
        "lambda x: x"
      ],
    "length": 5000
  },
  "reservoir": {
    "start_node": 95,
    "end_node": 100,
    "step": 2,
    "degree_function": "lambda x: np.sqrt(x)",
    "sigma": 0.5,
    "bias": 1,
    "leakage_rate": 0.3,
    "regression_parameter": 1e-8
  },
  "output": {
    "nodes": 2
  },
  "training": {
    "init": 1000,
    "train": 3000,
    "test": 2000,
    "error": 1000
  }
}"""


class Reservoir:
    def __init__(self):
        config_file_name = 'reservoir.config'
        global config
        if os.path.isfile(config_file_name):
            with open(config_file_name) as config_file:
                config = json.load(config_file, object_pairs_hook=OrderedDict)
        else:
            config = json.loads(default_config, object_pairs_hook=OrderedDict)
            print('Config file not exist, using default config instead!')

        # Input layer
        self.M = config["input"]["nodes"]
        self.input_len = config["input"]["length"]
        self.input_func = []
        dataset = []
        for i in range(self.M):
            self.input_func.append(eval(config["input"]["functions"][i]))
            dataset.append(self.input_func[i](
                np.arange(self.input_len) / self.input_len))
        self.dataset = np.array(list(zip(*dataset))).T # shape = (M, length)

        # Reservoir layer
        self.start_node = config["reservoir"]["start_node"]
        self.N = self.start_node
        self.step = config["reservoir"]["step"]
        self.end_node = config["reservoir"]["end_node"]
        self.degree_func = eval(config["reservoir"]["degree_function"])
        self.D = self.degree_func(self.start_node)
        self.sigma = config["reservoir"]["sigma"]
        self.bias = config["reservoir"]["bias"]
        self.alpha = config["reservoir"]["leakage_rate"]
        self.beta = config["reservoir"]["regression_parameter"]

        # Output layer
        self.P = config["output"]["nodes"]

        # Training relevant
        self.init_len = config["training"]["init"]
        self.train_len = config["training"]["train"]
        self.test_len = config["training"]["test"]
        self.error_len = config["training"]["error"]


    def train(self):
        # collection of reservoir state vectors
        self.R = np.zeros(
            (1 + self.N + self.M, self.train_len - self.init_len))
        # collection of input signals
        self.S = np.vstack((x[self.init_len + 1: self.train_len + 1] for x in self.dataset))
        self.r = np.zeros((self.N, 1))
        np.random.seed(42)
        self.Win = np.random.uniform(-self.sigma,
                                     self.sigma, (self.N, self.M + 1))
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        g = nx.erdos_renyi_graph(self.N, self.D / self.N, 42, True)
        # nx.draw(g, node_size=self.N)
        self.A = nx.adjacency_matrix(g).todense()
        # spectral radius: rho
        self.rho = max(abs(np.linalg.eig(self.A)[0]))
        self.A *= 1.25 / self.rho
        # run the reservoir with the data and collect r
        for t in range(self.train_len):
            u = np.vstack((x[t] for x in self.dataset))
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,
                                                                             self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            if t >= self.init_len:
                self.R[:, [t - self.init_len]
                       ] = np.vstack((self.bias, u, self.r))[:, 0]
        # train the output
        R_T = self.R.T  # Transpose
        # Wout = (s * r^T) * ((r * r^T) + beta * I)
        self.Wout = np.dot(np.dot(self.S, R_T), np.linalg.inv(
            np.dot(self.R, R_T) + self.beta * np.eye(self.M + self.N + 1)))

    def _run(self):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        self.S = np.zeros((self.P, self.test_len))
        u = np.vstack((x[self.train_len] for x in self.dataset))
        for t in range(self.test_len):
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,
                                                                             self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            s = np.dot(self.Wout, np.vstack((self.bias, u, self.r)))
            self.S[:, t] = np.squeeze(np.asarray(s))
            # use output as input
            u = s
        # compute Root Mean Square (RMS) error for the first self.error_len time steps
        self.RMS = []
        for i in range(self.P):
          self.RMS.append(sum(np.square(
            self.dataset[i, self.train_len+1: self.train_len+self.error_len+1] - self.S[i, 0: self.error_len])) / self.error_len)

    def draw(self):
      plt.subplots(1, self.M)
      plt.suptitle('N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
      for i in range(self.M):
        ax = plt.subplot(1, self.M, i + 1)
        plt.text(0.5, -0.1, 'RMS = %.15e' % self.RMS[i], size=10, ha="center", transform=ax.transAxes)
        plt.plot(self.S[i], label = 'prediction')
        plt.plot(self.dataset[i][self.train_len + 1 : self.train_len + self.test_len + 1], label = 'input signal')
        plt.title(config["input"]["functions"][i])
        plt.legend(loc = 'upper right')
        # plt.savefig('N = ' + str(self.N), dpi = 300)
      plt.show()
    
    def run(self):
        with open('reservoir.output', 'a') as output:
            prompt = '# ' + str(datetime.datetime.now()) + \
                '\n' + json.dumps(config, indent=4) + '\n'
            print(prompt, end='')
            output.write(prompt)
            for i in range(self.start_node, self.end_node + 1, self.step):
                self.N = i
                self.D = self.degree_func(self.N)
                self.train()
                self._run()
                for j in range(1):
                  res = 'N = ' + str(self.N) + ', D = ' + '%.15f' % self.D + \
                      ', RMS = ' + '%.15e' % Decimal(self.RMS[j]) + '\n'
                  print(res, end='')
                output.write(res)
                config["reservoir"]["start_node"] = i
                self.draw()


# Invoke automatically when exit, write the progress back to config file
def exit_handler():
    global config
    with open('reservoir.config', 'w') as config_file:
        config_file.write(json.dumps(config, indent = 4))
    print('Program finished! Current node = ' +
          str(config["reservoir"]["start_node"]))


if __name__ == '__main__':
    atexit.register(exit_handler)
    r = Reservoir()
    r.run()
