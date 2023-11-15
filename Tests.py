import json
import pickle
import random
from itertools import islice

import networkx as nx
import matplotlib.pyplot as plt


#this approach eliminates links from the network

graph = pickle.load(open('small_network.pickle', 'rb'))
G = nx.Graph(graph)
print("\n Original", G)
print("nodes: ", G.nodes())
print("edges: ", G.edges())
plt.subplot(1,3,1)
plt.title("Original")
nx.draw(G, with_labels=True)

H = nx.barabasi_albert_graph(25, 2)
print("\n barabasi_albert_graph", H)
print("nodes: ", H.nodes())
print("edges: ", H.edges())
plt.subplot(1,3,2)
plt.title("barabasi_albert_graph")
nx.draw(H, with_labels=True)

I = nx.erdos_renyi_graph(25,0.2)
print("\n erdos_renyi_graph", I)
print("nodes: ", I.nodes())
print("edges: ", I.edges())
plt.subplot(1,3,3)
plt.title("erdos_renyi_graph")
nx.draw(I, with_labels=True)

#G.remove_edge
#G.remove_node()

#pickle.dump(G, open("network_edges_change.pickle", "wb"))


plt.show()