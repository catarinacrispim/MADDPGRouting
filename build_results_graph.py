import json
import pickle
import random
from itertools import islice
import numpy as np
from environmental_variables import PATH_SIMULATION

import networkx as nx
import matplotlib.pyplot as plt
import datetime

day = datetime.date.today().day
month = datetime.date.today().month
hh = datetime.datetime.now().hour
mm = datetime.datetime.now().minute

BATCH_DATA = True
batch_size = 3
topology =""
epochs = ""

#Get data from files
#"Local critic, Duelling Q Network"
file1=""
data1 = np.loadtxt(file1, delimiter=',', dtype=float)

#Central critic, Simple Q Network"
file2=""
data2 =  np.loadtxt(file2, delimiter=',', dtype=float)

#Central critic, Duelling Q Network"
file3=""
data3 =  np.loadtxt(file3, delimiter=',', dtype=float)


if not BATCH_DATA:
    plt.plot(data1[1,:], label = "Local critic, Dueling Q Network")
    plt.plot(data2[1,:], label = "Central critic, Simple Q Network")
    plt.plot(data3[1,:], label = "Central critic, Dueling Q Network")
else:
    aux1 = data1[1,:]
    batch_data1 = [sum(aux1[i:i+batch_size]) for i in range(0, len(aux1), batch_size)]
    aux2 = data2[1,:]
    batch_data2 = [sum(aux2[i:i+batch_size]) for i in range(0, len(aux2), batch_size)]
    aux3 = data3[1,:]
    batch_data3 = [sum(aux3[i:i+batch_size]) for i in range(0, len(aux3), batch_size)]
    plt.plot(batch_data1, label = "Local critic, Dueling Q Network")
    plt.plot(batch_data2, label = "Central critic, Simple Q Network")
    plt.plot(batch_data3, label = "Central critic, Dueling Q Network")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.title(f"Rewards per epoch in training")

plt.savefig(f"{PATH_SIMULATION}/results/results_{topology}_{epochs}epochs_{batch_size}batch_{day}-{month}_{hh}:{mm}.png")

plt.show()