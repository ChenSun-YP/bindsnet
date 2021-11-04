import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)
import numpy as np
import data
# from torch import nn
# from torch.utils.data import DataLoader
# import torchaudio
# from torchaudio import datasets
# from torchvision.transforms import ToTensor, Lambda, Compose
# import matplotlib.pyplot as plt
# import multiprocessing
# import os
# import pandas as pd
# from bindsnet.network import Network
# from bindsnet.pipepyline import EnvironmentPipeline
from bindsnet.encoding import bernoulli
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes
# from bindsnet.pipeline.action import select_softmax
# import matplotlib.pyplot as plt
# from IPython.display import Audio, display
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from bindsnet import encoding
from bindsnet.network import Network, nodes, topology, monitors
from bindsnet.pipeline import EnvironmentPipeline


if __name__ =="__main__":


  # dataset = data.get_gtzan(subset='validation') # get dataset

  print('startloder')
  data = data.get_gtzan()
  train_loader = DataLoader(data, batch_size=10, shuffle=True)
  
  network = Network(dt=1.0)  # Instantiates network.
  
  X = nodes.Input(100)  # Input layer.
  Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
  C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

  # Spike monitor objects.
  M1 = monitors.Monitor(obj=X, state_vars=['s'])
  M2 = monitors.Monitor(obj=Y, state_vars=['s'])

  # Add everything to the network object.
  network.add_layer(layer=X, name='X')
  network.add_layer(layer=Y, name='Y')
  network.add_connection(connection=C, source='X', target='Y')
  network.add_monitor(monitor=M1, name='X')
  network.add_monitor(monitor=M2, name='Y')

  # Create Poisson-distributed spike train inputs.
  # data = dataset[0][0]  # Generate random Poisson rates for 100 input neurons.
  data = torch.sigmoid(dataset[0][0])
  print('startencoding')
  train = encoding.poisson(datum=data, time=50)  # Encode input as 5000ms Poisson spike trains.
  print('startsimulation')

  # Simulate network on generated spike trains.
  inputs = {'X': train}  # Create inputs mapping.
  network.run(inputs=inputs, time=50)  # Run network simulation.

  # Plot spikes of input and output layers.
  spikes = {'X': M1.get('s'), 'Y': M2.get('s')}

  fig, axes = plt.subplots(2, 1, figsize=(12, 7))
  for i, layer in enumerate(spikes):
    axes[i].matshow(spikes[layer], cmap='binary')
    axes[i].set_title('%s spikes' % layer)
    axes[i].set_xlabel('Time');
    axes[i].set_ylabel('Index of neuron')
    axes[i].set_xticks(());
    axes[i].set_yticks(())
    axes[i].set_aspect('auto')

  plt.tight_layout();
  plt.show()

  network = Network(dt=1.0)
  # Layers of neurons.
  inpt = Input(n=80 * 80, shape=[80, 80], traces=True)
  middle = LIFNodes(n=100, traces=True)
  out = LIFNodes(n=4, refrac=0, traces=True)
  # Connections between layers.
  inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
  middle_out = Connection(source=middle, target=out, wmin=0, wmax=1)
  # Add all layers and connections to the network.
  network.add_layer(inpt, name="Input Layer")
  network.add_layer(middle, name="Hidden Layer")
  network.add_layer(out, name="Output Layer")
  network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
  network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

  n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

  # Neuron assignments and spike proportions.
  n_classes = 10
  assignments = -torch.ones(n_neurons)
  proportions = torch.zeros(n_neurons, n_classes)
  rates = torch.zeros(n_neurons, n_classes)
  network.train()

  # Train the network.
  print("\nBegin training.\n")
  start = t()

  







