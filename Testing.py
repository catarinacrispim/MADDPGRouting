import json
import os
import pickle
import random
from itertools import islice

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


data_file_path = "/home/student/results/data_while_training.csv"
if os.path.exists(data_file_path):
    existing = np.loadtxt(data_file_path, delimiter=',')
    existing_rewards = existing[1,:]
    existing_epochs = existing[0,:]
else:
    existing_rewards = np.array([])
    existing_epochs = np.array([])

y = [5,6,7]
x= np.arange(0, 7)
    
#y = [1,2,3,4]
#x = np.arange(0, len(y))

print(len(y))

combining_rewards = np.concatenate([existing_rewards, y])
#combining_epochs = np.concatenate([existing_epochs, x])

np.savetxt(data_file_path, (x, combining_rewards), delimiter=',')