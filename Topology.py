import json
import pickle
import random
from itertools import islice

import networkx as nx
import matplotlib.pyplot as plt

G = pickle.load(open('small_network.pickle', 'rb'))
#G = pickle.load(open('topology_arpanet.pickle', 'rb'))
#G = pickle.load(open('service_provider_network.pickle', 'rb'))

#TOPOLOGY_FILE_NAME = "/home/student/MADDPGRouting/topology_arpanet.txt"
#TOPOLOGY_FILE_NAME = "/home/student/MADDPGRouting/topology.txt"
#G = nx.Graph()

"""with open(TOPOLOGY_FILE_NAME, 'r') as topo:
    for line in topo.readlines():
        data = line.split()
        nodes = line.split()[:2]
        for node in nodes:
            if node[0] == 'S':
                #node = f"H{node[1:]}"
                node = int(node[1:]) - 1
            #elif node [0] == 'H':
            #    node = int(node[1:]) -1 +20 #for arpanet
                if not G.has_node(node):
                    G.add_node(node)
        
        #bw = int(data[2])/10
        bw = int(data[2])

        if nodes[0][0] == 'S':
                #nodes[0] = f"H{nodes[0][1:]}"
                nodes[0] = int(nodes[0][1:]) - 1
        else:
             continue
        #elif nodes[0][0] == 'H':
        #        nodes[0] = int(nodes[0][1:]) -1 +20 #for arpanet
        
        if nodes[1][0] == 'S':
                #nodes[1] = f"H{nodes[1][1:]}"
                nodes[1] = int(nodes[1][1:]) - 1
        else: 
            continue
        #elif nodes[1][0] == 'H':
        #        #nodes[1] = f"H{nodes[1][1:]}"
        #        nodes[1] = int(nodes[1][1:]) - 1 + 20 #arpanet
        
        G.add_edge(nodes[0], nodes[1], bw = int(bw))"""

degrees = G.degree()
max_degree = max(degrees, key=lambda x: x[1]) 

print("\n nodes: ", G.nodes())
print("\n edges: ", G.edges(data=True))
print("\n Original", G)
print("\nmax number of connections: ", max_degree[1])
nx.draw(G, with_labels=True)

#pickle.dump(G, open("topology_arpanet.pickle", "wb"))

plt.show()

