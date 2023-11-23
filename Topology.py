import json
import pickle
import random
from itertools import islice

import networkx as nx
import matplotlib.pyplot as plt


"""def add_switch(self, name):
    if name not in self.network.keys():
        self.switches[name.replace("S", "")] = name
        self.network.addSwitch(name)

def add_link(self, source, destination, link_options):
    if not self.network.linksBetween(self.network.get(source), self.network.get(destination)):
        self.network.addLink(self.network.get(
            source), self.network.get(destination), **link_options) """ 


#graph = pickle.load(open('small_network.pickle', 'rb'))
TOPOLOGY_FILE_NAME = "/home/student/MADDPGRouting/topology.txt"
G = nx.Graph()

with open(TOPOLOGY_FILE_NAME, 'r') as topo:
    for line in topo.readlines():
        data = line.split()
        nodes = line.split()[:2]
        for node in nodes:
            if node[0] == 'S':
                #node = f"H{node[1:]}"
                node = int(node[1:]) - 1
            else:
                 continue
            if not G.has_node(node):
                G.add_node(node)
        bw = int(data[2])/10
        if nodes[0][0] == 'S':
                #nodes[0] = f"H{nodes[0][1:]}"
                nodes[0] = int(nodes[0][1:]) - 1
        else:
             continue
        if nodes[1][0] == 'S':
                #nodes[1] = f"H{nodes[1][1:]}"
                nodes[1] = int(nodes[1][1:]) - 1
        else: 
            continue
        G.add_edge(nodes[0], nodes[1], bw = int(bw))

# from networkaigym "containernet_api_topo.py"
"""with open("/home/student/MADDPGRouting/topology.txt", 'r') as topology:
    for line in topology.readlines():
        cols = line.split()
        for node in cols[:2]:
            if node[0] == 'S':
                add_switch(node)
            else:
                add_host(node, container_params)

        link_bw = int(cols[2])
        if len(cols) < 4:
            link_options = dict(bw=link_bw)
        else:
            link_options = dict(
                bw=link_bw, delay=f'{cols[3]}ms', loss=float(cols[4]))
        add_link(cols[0], cols[1], link_options)
        bw_capacity[(cols[0], cols[1])] = link_bw
        bw_capacity[(cols[1], cols[0])] = link_bw"""

print("\n nodes: ", G.nodes())
print("\n edges: ", G.edges(data=True))
print("\n Original", G)
nx.draw(G, with_labels=True)

#pickle.dump(G, open("intranet_network.pickle", "wb"))

plt.show()

