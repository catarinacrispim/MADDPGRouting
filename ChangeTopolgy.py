import json
import pickle
import random
from itertools import islice

import networkx as nx
import matplotlib.pyplot as plt


#this approach eliminates links from the network

graph_to_change = pickle.load(open('small_network.pickle', 'rb'))
G = nx.Graph(graph_to_change)
print("\n Original", G)
print("nodes: ", G.nodes())
print("edges: ", G.edges())

#for edge in G.edges(data=True):
#    print(f"\n edge: {[edge[:2]]}, data {[edge[2]]}")

plt.subplot(1,2,1)
plt.title("Original")
nx.draw(G, with_labels=True)

high_degree_nodes = [node for node in G.nodes() if G.degree(node)>1]
#edges = list(G.edges())
edges = [edge for node in high_degree_nodes for edge in G.edges(node)]

remove = random.sample(edges, min(5, len(edges)))

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

#pickle.dump(G, open("network_edges_change.pickle", "wb"))

plt.show()