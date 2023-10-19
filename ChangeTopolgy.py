import json
import pickle
import random
from itertools import islice

import networkx as nx
import matplotlib.pyplot as plt


#this approach eliminates routers from the network

graph_to_change = pickle.load(open('small_network.pickle', 'rb'))
G = nx.Graph(graph_to_change)
print("\n Original", G)
print("nodes: ", G.nodes())
print("edges: ", G.edges())

plt.subplot(1,2,1)
plt.title("Original")
nx.draw(G, with_labels=True)

edges = list(G.edges())
remove = random.sample(edges, 4)

for edge in remove:
    print("removing edge: ", edge)
    G.remove_edge(*edge)

print("\n Modified", G)
print("nodes: ", G.nodes())
print("edges: ", G.edges())
plt.subplot(1,2,2)
plt.title("Modified")
nx.draw(G, with_labels=True)

#G.remove_edge
#G.remove_node()

pickle.dump(G, open("network_edges_change.pickle", "wb"))

plt.show()