import json
import pickle
import random
from itertools import islice
import numpy as np
from environmental_variables import PATH_SIMULATION


import networkx as nx
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd


day = datetime.date.today().day
month = datetime.date.today().month
hh = datetime.datetime.now().hour
mm = datetime.datetime.now().minute

#graph_x_axis = np.arange(0, episode_size)
 
#Get data from files
file1=""
data1 = pd.read_csv(file1)

file2=""
data2 = pd.read_csv(file2)

file3=""
data3 = pd.read_csv(file3)


plt.plot(data1[0], data1[1], label = "Central critic, Duelling Q Network")
plt.plot(data2[0], data2[1], label = "Central critic, Simple Q Network")
plt.plot(data3[0], data3[1], label = "Local critic, Duelling Q Network")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.title(f"Rewards per epoch in training")

plt.savefig(f"/home/{PATH_SIMULATION}/results/results_{day}-{month}_{hh}:{mm}.png")

#{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{learning}.png")
#np.savetxt(f"/home/{PATH_SIMULATION}/results/{NR_EPOCHS}epochs_{EPOCH_SIZE}episodes_{CRITIC_DOMAIN}_{NEURAL_NETWORK}_{TOPOLOGY_TYPE}_{learning}_{day}-{month}_{hh}:{mm}/data.csv", (graph_x_axis, graph_y_axis), delimiter=',')
plt.show()