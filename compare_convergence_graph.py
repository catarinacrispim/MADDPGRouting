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

BATCH_DATA = False
batch_size = 3
topology ="internet"
epochs = 20
algorithm="central_simple"

#Get data from files
#"Local critic, Duelling Q Network"
file1="/home/student/results/5epochs_20episodes_central_critic_simple_q_network_internet_train_18-3_10:2/data_total.csv"
data1 = np.loadtxt(file1, delimiter=',', dtype=float)

#Central critic, Simple Q Network"
file2="/home/student/results/5epochs_20episodes_central_critic_simple_q_network_internet_test_and_train_remove_edges_18-3_10:19/data_total.csv"
data2 =  np.loadtxt(file2, delimiter=',', dtype=float)


if not BATCH_DATA:
    plt.plot(data1[1,:epochs], label = "training")
    plt.plot(data2[1,:epochs], label = "training changed topology")
else:
    aux1 = data1[1,:epochs]
    batch_data1 = [sum(aux1[i:i+batch_size]) for i in range(0, len(aux1), batch_size)]
    aux2 = data2[1,:epochs]
    batch_data2 = [sum(aux2[i:i+batch_size]) for i in range(0, len(aux2), batch_size)]
    plt.plot(batch_data1, label = "Training")
    plt.plot(batch_data2, label = "Training changed topology")

plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Reward")
plt.title(f"Rewards per epoch in training")

if BATCH_DATA == True:
    plt.savefig(f"{PATH_SIMULATION}/results/results_{topology}_{algorithm}_{epochs}epochs_{batch_size}batch_{day}-{month}_{hh}:{mm}.png")
else:
    plt.savefig(f"{PATH_SIMULATION}/results/results_{topology}_{algorithm}_{epochs}epochs_{day}-{month}_{hh}:{mm}.png")

plt.show()